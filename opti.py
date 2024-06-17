import torch
import ULA.pot as pot
import ULA.geo as geo
from math import *

def compute_grad(U,z_line,theta):
    print("inside compute_grad z",z_line)
    pot = U(z_line,theta)
    pot.backward()
    return torch.tensor([z_line.grad,theta.grad])
    

def compute_inv_hessian(U,z,theta):

    theta = theta.clone().detach().requires_grad_(True)

    U_theta= lambda thet:U(z,thet)
    hessian = torch.autograd.functional.hessian(U_theta,theta)
    inv_hess = torch.linalg.inv(hessian + 10e-10 * torch.eye(len(theta)))
    return inv_hess

#z_list -> (len(z_line) , nb_particules) tensor

class ULA:
    def __init__(self,theta_init):
        self.theta = theta_init
        self.z_list = torch.tensor([])
    
    def init_z(self,nb_part,nb_latent,f_init):
        self.z_list = torch.zeros(nb_part,nb_latent)
        for i in range(nb_part):
            z_alea = f_init(self.theta)
            self.z_list[i] = z_alea.clone().detach()

    def ULA_descente_step(self,U,var_bruit,stepsize):

        nb_part = len(self.z_list)
        nb_latent =len(self.z_list[0])
        nb_param = len(self.theta)
        std = torch.sqrt(torch.tensor(var_bruit))

        noise_z = torch.reshape(torch.normal(torch.zeros(nb_part*nb_latent) , torch.fill(torch.zeros(nb_part*nb_latent),std)),(-1,nb_latent))
        noise_theta = torch.normal(torch.zeros(len(self.theta)),torch.fill(torch.zeros(len(self.theta)),std))

        #potentielle erreur vmap (calling .item())
        grad_z_theta = torch.zeros((nb_part,nb_latent + nb_param))
        for i in range(nb_part):
            z_theta = (torch.cat((self.z_list[i],self.theta))).clone().detach().requires_grad_(True)
            p = U(z_theta[:nb_latent],z_theta[nb_latent:])
            p.backward()
            grad_z_theta[i]=z_theta.grad

        grad_z = grad_z_theta[:,:nb_latent]
        grad_theta = grad_z_theta[:,nb_latent:]
        #print("z_theta",z_theta)
        #print("grad_z_theta",grad_z_theta)
        #print("grad_z",grad_z)
        #print("grad_theta",grad_theta)


        grad_sum = torch.sum(grad_theta,0)
        #print("grad_sum",grad_sum)
        
        #on prend la hessienne par rapport à un des z arbitrairement, on pourrait prendre la moyenne sur tous les z des matrices de conditionnement 
        cond_mat = torch.abs(compute_inv_hessian(U,self.z_list[0],self.theta))
        sqrt_cond_mat= torch.sqrt(cond_mat) #il faudrait prendre la racine carrée au sens de transposee_R * R = A mais ici la hessienne est diagonale
        #print("sqrt_cond_mat",sqrt_cond_mat)

        update_theta = sqrt(2*stepsize/nb_part)*torch.linalg.matmul(sqrt_cond_mat,noise_theta) - (stepsize/nb_part) * torch.linalg.matmul(cond_mat,grad_sum)
        #print("update_theta",update_theta)
        #sans conditionnement:
        #update_theta = sqrt(2*stepsize/nb_part)*torch.normal(torch.zeros(1),std) - (stepsize/nb_part) * grad_sum

        update_z = sqrt(2*stepsize)*noise_z - stepsize*grad_z

        self.theta = self.theta + update_theta
        self.z_list = self.z_list + update_z

def compute_f_init(dim_geo,dim_pca,nb_patients):

    def f_init(theta):
        nb_pop_param = 1 + 2*dim_geo + (dim_geo-1)*dim_pca
        z = torch.zeros(nb_pop_param  + (2+dim_pca)*nb_patients)
        z[:nb_pop_param ] = theta[:nb_pop_param ]
        std_tau = torch.sqrt(torch.exp(theta[nb_pop_param]))
        std_xi = torch.sqrt(torch.exp(theta[nb_pop_param+1]))

        xi_init = torch.normal(torch.zeros(nb_patients),torch.fill(torch.zeros(nb_patients),std_xi))
        tau_init = torch.normal(torch.zeros(nb_patients),torch.fill(torch.zeros(nb_patients),std_tau))
        s_init = torch.normal(torch.zeros(nb_patients*dim_pca),torch.ones(nb_patients*dim_pca))

        init_ind_line = torch.cat((tau_init,xi_init,s_init))
        z_rec = torch.transpose(torch.reshape(init_ind_line,(-1,nb_patients)),0,1)
        z_line = torch.reshape(z_rec,(-1,))

        z[nb_pop_param:] = z_line

        return z
    
    return f_init

def ULA_algo(time,y,mask,exp_para,theta_init,nb_iter,nb_part,stepsizes,var_bruit,var_param,dim_pca):
    model = ULA(theta_init)
    dim_geo = len(y[0][0])
    nb_patients = len(time)

    f_init = compute_f_init(dim_geo,dim_pca,nb_patients)
    nb_latent = (2+dim_pca)*nb_patients + 1+2*dim_geo + (dim_geo-1)*dim_pca
    model.init_z(nb_part,nb_latent,f_init)

    for i in range(nb_iter):
        U = pot.potentiel(exp_para,time,y,mask,var_param,dim_geo,dim_pca)
        model.ULA_descente_step(U,var_bruit[i],stepsizes[i])
        if(i%100 == 0):
            print("step ",i)
            print(model.theta)

    return model.theta
    
def generate_data(t, theta, var_param , exp_para , dim_geo, dim_pca):
    nb_patients = len(t)
    nb_max_measures = len(t[0])
    nb_pop_param = 2*dim_geo + (dim_geo-1)*dim_pca + 1
    param_metric = theta[nb_pop_param+3:]

    t0 = torch.normal(theta[0],torch.tensor([var_param[0]]))
    p0 = torch.normal(theta[1:dim_geo+1],torch.fill(torch.zeros(dim_geo),var_param[1]))
    v0 = torch.normal(theta[dim_geo+1:2*dim_geo+1],torch.fill(torch.zeros(dim_geo),var_param[2]))
    std_tau = torch.sqrt(torch.exp(theta[nb_pop_param]))
    std_xi = torch.sqrt(torch.exp(theta[nb_pop_param+1]))
    std_eps = torch.sqrt(torch.exp(theta[nb_pop_param+2]))
    xi = torch.normal(torch.zeros(nb_patients),torch.fill(torch.zeros(nb_patients),std_xi))
    tau = torch.normal(torch.zeros(nb_patients),torch.fill(torch.zeros(nb_patients),std_tau))
    eps = torch.normal(torch.zeros((nb_patients,nb_max_measures,dim_geo)),torch.fill(torch.zeros((nb_patients,nb_max_measures,dim_geo)),std_eps))

    if(dim_pca > 0):
        beta = torch.normal(theta[2*dim_geo+1:nb_pop_param].reshape(dim_geo-1,dim_pca),torch.fill(torch.zeros((dim_geo-1)*dim_pca),var_param[3]).reshape(dim_geo-1,dim_pca))
        base = pot.derive_ortho_base_v0(v0,dim_geo)
        #print("beta",beta)
        #print("base",base)
        s = torch.normal(torch.zeros((nb_patients,dim_pca)),torch.ones((nb_patients,dim_pca)))
        w = pot.compute_w_from_s(s,beta,base)
    else:
        w = torch.zeros((nb_patients,dim_geo))
    z_rec = torch.zeros((nb_patients,dim_pca+2))
    z_rec[:,0] = tau
    z_rec[:,1] = xi
    altered_time = pot.time_param(t,torch.ones_like(t),z_rec,t0)
    #print("altered_time",altered_time)
    y = torch.vmap(lambda w_i,t_i:exp_para(param_metric,p0,v0,w_i,t_i))(w,altered_time)
    return y+eps

def exp_para_lin(param_metric,p0,v0,w_i,t_i):
    exp_para = torch.vmap(lambda t:p0+w_i+v0*t)(t_i)
    return exp_para

def cree_theta(t0,p0,v0,beta,log_var_tau,log_var_xi,log_var_eps,param_metric,dim_geo,dim_pca):
    theta = torch.zeros(4 + 2*dim_geo + (dim_geo-1)*dim_pca + len(param_metric))
    theta[0] = t0
    theta[1:dim_geo+1]=p0
    theta[dim_geo+1 : 2*dim_geo+1]=v0
    theta[2*dim_geo+1: 2*dim_geo+1 + dim_pca*(dim_geo-1)]=beta.reshape((-1,))
    theta[2*dim_geo+1 + dim_pca*(dim_geo-1)] = log_var_tau
    theta[2*dim_geo+1 + dim_pca*(dim_geo-1) +1] = log_var_xi
    theta[2*dim_geo+1 + dim_pca*(dim_geo-1) + 2] = log_var_eps
    theta[2*dim_geo+1 + dim_pca*(dim_geo-1)+3:] = param_metric
    return theta


####################################        TESTS       ########################################################


#TEST DE LA DESCENTE AVEC UN MODELE SIMPLE (ok)

var = torch.tensor(.1)
nb_part = 10
nb_iter = 10000
stepsizes = torch.fill(torch.zeros(nb_iter),0.001) 
var_bruit = torch.fill(torch.zeros(nb_iter),0.000001)
time = torch.tensor([[1.,2.,3.,4.,5.],[4.,5.,6.,7.,8.]])

theta_data = torch.tensor([0.1,-2.])
bruit_data = torch.exp(theta_data[1])
print
y = time * theta_data[0] + torch.normal(torch.zeros_like(time),torch.fill(torch.zeros_like(time),bruit_data))

def potentiel_gaussien(var,y,time):
    def U(z,theta):
        prior = ((z[0]-theta[0])**2)/2*var + torch.log(var)/2
        eval = z[0]*time
        posterior = torch.sum(((y-eval)**2)/(2*torch.exp(theta[1]))) + theta[1]/2
        return prior + posterior
    return U

def ULA_gaussien(y,time,nb_part,nb_iter,stepsizes,var_bruit,var):
    model = ULA(theta_data)
    model.z_list = torch.tensor([[1.],[0.],[3.]])
    model.theta = torch.tensor([1.5,1.])
    for i in range(nb_iter):
        U = potentiel_gaussien(var,y,time)
        model.ULA_descente_step(U,var_bruit[i],stepsizes[i])
        if (i%100 ==0):
           print("step",i)
           print(model.theta)

#ULA_gaussien(y,time,10,nb_iter,stepsizes,var_bruit,var)

dim_geo =1
dim_pca = 0
nb_patients = 60
nb_mesures = 5

t = torch.exp(torch.normal(torch.zeros(nb_patients*nb_mesures),torch.fill(torch.zeros(nb_patients*nb_mesures),.5)).reshape(nb_patients,nb_mesures))
t0 = 0.
p0 = torch.tensor([0.3])
v0 = torch.tensor([1.])
beta = torch.tensor([[]])
log_var_tau = 1.
log_var_xi = -1.
log_var_eps = -1.
param_metric = torch.tensor([1.,0.,2.])
theta = cree_theta(t0,p0,v0,beta,log_var_tau,log_var_xi,log_var_eps,param_metric,dim_geo,dim_pca)
var_param = torch.tensor([.001,.001,.001,.001])
y = generate_data(t,theta,var_param,geo.exp_para_poly1D,dim_geo,dim_pca)

theta_init = torch.tensor([1.,.2,1.5,-1.,-1.,-1.,1.,0.,2])
#theta_init = theta + torch.normal(torch.zeros_like(theta),torch.ones_like(theta)/2.)
nb_iter = 20000
stepsizes = torch.fill(torch.zeros(nb_iter),0.001) 
var_bruit = torch.fill(torch.zeros(nb_iter),0.0001)
nb_part = 1
resultat = ULA_algo(t,y,torch.ones_like(t),geo.exp_para_poly1D,theta_init,nb_iter,nb_part,stepsizes,var_bruit,var_param,dim_pca)

import torch
import ULA.pot as pot
import ULA.geo as geo
from timeit import default_timer
import matplotlib.pyplot as plt
from math import *
import numpy as np
import pandas as pd

tab_loss = [0.]

def compute_grad(U,z_line,theta):
    #print("inside compute_grad z",z_line)
    pot = U(z_line,theta)
    pot.backward()
    return torch.tensor([z_line.grad,theta.grad])
    
#z_list -> (len(z_line) , nb_particules) tensor

class ULA:
    def __init__(self,theta_init,is_neural,conditionnement):
        self.theta = theta_init
        self.z_list = torch.tensor([])
        self.is_neural = is_neural
        self.conditionnement = conditionnement
        self.nb_tour = 0
        self.cond = torch.zeros(len(theta_init))
        self.z_moyenne = torch.tensor([])
    
    def init_z(self,nb_part,nb_latent,f_init):
        self.z_list = torch.zeros(nb_part,nb_latent)
        for i in range(nb_part):
            z_alea = f_init(self.theta)
            self.z_list[i] = z_alea.clone().detach()

    def compute_inv_hessian(self,U,z,theta):
       theta = theta.clone().detach().requires_grad_(True)
       U_theta= lambda thet:U(z,thet,tab_loss,False)
       hessian = torch.autograd.functional.hessian(U_theta,theta) 

       if self.is_neural :
           loss = U_theta(theta)
           loss.backward(create_graph = True)
           grad = torch.autograd.grad(loss, neural_net.parameters(), create_graph=True)
           grad = torch.cat([g.reshape(-1) for g in grad])
           hessian_metric = []
           for g in grad.view(-1):
               grad2 = torch.autograd.grad(g, neural_net.parameters(), create_graph=True)
               hessian_metric.append(torch.cat([torch.flatten(g2_param) for g2_param in grad2]).unsqueeze(0))

           hessian_metric = torch.cat(hessian_metric, dim=0)
           mini = torch.min(hessian_metric[hessian_metric != 0])
           print(mini)

           for i in range(len(hessian_metric)):
               if(hessian_metric[-(i+1)][-(i+1)] !=0):
                   hessian[-(i+1)][-(i+1)] = hessian_metric[-(i+1)][-(i+1)]
               else:
                   hessian[-(i+1)][-(i+1)] = 1000
        
        
       inv_hess = 1/hessian
       return inv_hess

    def ULA_descente_step(self,U,var_bruit,stepsize,neural_net=0):

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
            p = U(z_theta[:nb_latent],z_theta[nb_latent:],tab_loss,True)
            p.backward(retain_graph =True)             
            #p.backward()
            grad_z_theta[i]=z_theta.grad
            if(self.is_neural):
                grad_z_theta[i][nb_latent+4+2*dim_geo+(dim_geo-1)*dim_pca:] = neural_net.get_grad()

        grad_z = grad_z_theta[:,:nb_latent]
        grad_theta = grad_z_theta[:,nb_latent:]
        #print("z_theta",z_theta)
        #print("grad_z_theta",grad_z_theta)
        #print("grad_z",grad_z)
        #print("grad_theta",grad_theta)


        grad_sum = torch.sum(grad_theta,0)
        #print("grad_sum",grad_sum)
         
        update_theta_grad = - (stepsize/nb_part) * grad_sum
        update_theta_noise = sqrt(2*stepsize/nb_part)*noise_theta

        #on prend la hessienne par rapport à un des z arbitrairement, on pourrait prendre la moyenne sur tous les z des matrices de conditionnement 
        if self.conditionnement:
            if self.is_neural :
                if (self.nb_tour % 50000== 0):
                    cond_mat = torch.abs(self.compute_inv_hessian(U,self.z_list[0],self.theta))
                    for i in range(nb_param):
                        self.cond[i] = cond_mat[i][i]
                        maxi = torch.max(self.cond[self.cond!=0])
                        self.cond /= maxi
            else:
                if (self.nb_tour % 200== 0):
                    cond_mat = torch.abs(self.compute_inv_hessian(U,self.z_list[0],self.theta))
                    for i in range(nb_param):
                        self.cond[i] = cond_mat[i][i]
                        maxi = torch.max(self.cond[self.cond!=0])
                        self.cond /= maxi

            #print("sqrt_cond_mat",sqrt_cond_mat)
            for i in range(nb_param):
                update_theta_grad[i] = update_theta_grad[i] * self.cond[i]
                update_theta_noise[i] = update_theta_noise[i] * sqrt(self.cond[i])
        
        
        update_theta = update_theta_grad + update_theta_noise
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

def ULA_algo(time,y,mask,exp_para,theta_init,nb_iter,nb_part,stepsizes,var_bruit,var_param,dim_pca,is_neural,conditionnement,neural_net = 0):
    model = ULA(theta_init,is_neural,conditionnement)
    dim_geo = len(y[0][0])
    nb_patients = len(time)
    t_start = default_timer()

    f_init = compute_f_init(dim_geo,dim_pca,nb_patients)
    nb_latent = (2+dim_pca)*nb_patients + 1+2*dim_geo + (dim_geo-1)*dim_pca
    model.init_z(nb_part,nb_latent,f_init)

    liste_t =[]
    liste_theta = []

    for i in range(nb_iter):

        U = pot.potentiel(exp_para,time,y,mask,var_param,dim_geo,dim_pca)
        model.ULA_descente_step(U,var_bruit[i],stepsizes[i],neural_net)

        if(i%500 == 0):
            if(i==0):
                model.z_moyenne = torch.zeros_like(model.z_list)
            print("step ",i)
            print(model.theta)
            if( i%2000 == 0):
                liste_theta.append(model.theta)
                liste_t.append(default_timer() - t_start)
        model.nb_tour +=1
        model.z_moyenne = model.z_moyenne + 1/10*(model.z_list - model.z_moyenne)


    return liste_theta,liste_t,model.z_moyenne
    
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
    y = torch.vmap(lambda w_i,t_i:exp_para(param_metric,t0,p0,v0,w_i,t_i))(w,altered_time)
    return y+eps


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
        if (i%200 ==0):
           print("step",i)
           print("theta :",model.theta)

#ULA_gaussien(y,time,10,nb_iter,stepsizes,var_bruit,var)


dim_geo =1
dim_pca = 0
nb_patients = 1000
nb_mesures = 8
nb_iter = 200000
nb_part = 1
nb_neurones = 50
exp_para_net,neural_net = geo.neural_exp_para_func(dim_geo,nb_neurones)


t = torch.sort(torch.exp(torch.normal(torch.zeros(nb_patients*nb_mesures),torch.fill(torch.zeros(nb_patients*nb_mesures),.5)).reshape(nb_patients,nb_mesures)),1).values
t0 = 0.
p0 = torch.tensor([.1])
v0 = torch.tensor([.2])
beta = torch.tensor([[]])
log_var_tau = -1.
log_var_xi = -1.
log_var_eps = -3.
param_metric = torch.tensor([])


theta = cree_theta(t0,p0,v0,beta,log_var_tau,log_var_xi,log_var_eps,param_metric,dim_geo,dim_pca)
var_param = torch.tensor([.001,.001,.001,.001])
y = generate_data(t,theta,var_param,geo.exp_para_logistic,dim_geo,dim_pca)

modif_theta = torch.zeros_like(theta)
modif_theta[1] = .2
modif_theta[2] = -.1
theta_init = theta + modif_theta

stepsizes = torch.fill(torch.zeros(nb_iter),0.000005) 
var_bruit = torch.fill(torch.zeros(nb_iter),.001)
is_neural = False
conditionnement = True

tab_theta,tab_t = ULA_algo(t,y,torch.ones_like(t),geo.exp_para_logistic,theta_init,nb_iter,nb_part,stepsizes,var_bruit,var_param,dim_pca,is_neural,conditionnement)
theta_fin = theta_init
theta_fin = tab_theta[-1]

tab_t0 = tab_theta[:,0]
tab_p0 = tab_theta[:,1]
tab_v0 = tab_theta[:,2]
tab_eps = tab_theta[:,5]

tab_t = torch.tensor(tab_t)

tab_p0_modif = tab_p0 
plt.figure(1)
plt.plot(tab_t, 100 * torch.abs(tab_p0_modif - theta[1])/theta[1])
plt.title("erreur sur p0 en pourcentage au cours du temps")

plt.figure(2)
plt.plot(tab_t,100 * torch.abs(tab_v0 - 0.28)/0.28)
plt.title("erreur sur v0 en pourcentage au cours du temps")

plt.figure(3)
plt.plot(tab_t,tab_eps)
plt.title("logarithme du terme d'errreur (bruit du modèle)")

plt.show()



################################### TESTS DONNÉES RÉELLES ################

""" 
df = pd.read_csv('test.csv', delimiter=',')  
id_counts = df['ID'].value_counts()
valid_ids = id_counts[id_counts == 4].index
filtered_df = df[df['ID'].isin(valid_ids)]
df = filtered_df

id_tensor = torch.tensor(df['ID'].values, dtype=torch.float32)
time_tensor = torch.tensor(df['TIME'].values, dtype=torch.float32)
mes_mri_hippocampus_icv_tensor = torch.tensor(df['MES_MRI_HIPPOCAMPUS_ICV'].values, dtype=torch.float32)

 
dim_geo =1
dim_pca = 0
nb_patients = 672
nb_mesures = 4
nb_iter = 50000
nb_part = 5

t = time_tensor.reshape((-1,4))
y = mes_mri_hippocampus_icv_tensor .reshape((-1,4,1))


t0 = 80
p0 = torch.tensor([.2])
v0 = torch.tensor([.2])
beta = torch.tensor([[]])
log_var_tau = -1.
log_var_xi = -1.
log_var_eps = -1.
param_metric = torch.tensor([])



theta = cree_theta(t0,p0,v0,beta,log_var_tau,log_var_xi,log_var_eps,param_metric,dim_geo,dim_pca)
var_param = torch.tensor([.001,.001,.001,.001])

modif_theta = torch.zeros_like(theta)
theta_init = theta + modif_theta


stepsizes = torch.fill(torch.zeros(nb_iter),0.00002) 
var_bruit = torch.fill(torch.zeros(nb_iter),.0001)
is_neural = False
conditionnement = True

tab_theta,tab_t,z_moyenne = ULA_algo(t,y,torch.ones_like(t),geo.exp_para_lin,theta_init,nb_iter,nb_part,stepsizes,var_bruit,var_param,dim_pca,is_neural,conditionnement)
theta_fin = tab_theta[-1]


def observe_patient(theta_fin,t,z_moyenne,patient):
    t_patient = t[patient]
    t_patient_extend = torch.linspace(min(t_patient),max(t_patient),100)
    tau = z_moyenne[0][3 + 2*patient]
    xi = z_moyenne[0][4 + 2*patient]
    t0 = z_moyenne[0][0]
    altered_t_patient = np.exp(xi)*(t_patient_extend-t0-tau)+t0
    p0 = z_moyenne[0][1]
    v0 = z_moyenne[0][2]
    prediction = geo.exp_para_logistic(0,t0,p0,v0,0,altered_t_patient)
    reel = y[patient]
    plt.plot(t_patient_extend,prediction,label = "prédictions")
    plt.plot(t_patient,reel,'+',label = "mesures")
    plt.title("prédiction personalisées sur des données réelles")
    plt.xlabel("t")
    plt.ylabel("volume de l'hippocampe")
    plt.legend()



for i in range(20):
    plt.figure(i)
    observe_patient(theta_fin,t,z_moyenne,i)
    plt.show()

"""

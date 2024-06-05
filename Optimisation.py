import numpy as np
import torch
import scipy.integrate
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import torchode as to


class parametres:
    def __init__(self,t0_mean,p0_mean,v0_mean,var_xi,var_tau,var_eps):
        #paramètres géodésique moyenne
        self.t0_m = t0_mean
        self.p0_m = p0_mean
        self.v0_m = v0_mean

        #variance facteur d'acceleration
        self.log_var_xi = np.log(var_xi)
        #variance décalages temporels(age)
        self.log_var_tau =np.log(var_tau)
        #variance du bruit gaussien
        self.log_var_eps = np.log(var_eps)

    def show(self):
        print("t0_mean :" + str(self.t0_m) + "   |   p0_mean :" + str(self.p0_m) + "   |   v0_mean :" + str(self.v0_m) + "\n")
        print("var_xi :" + str(np.exp(self.log_var_xi)) + "   |   var_tau :" + str(np.exp(self.log_var_tau))  + "   |   var_eps :" + str(np.exp(self.log_var_eps))  + "\n")
        print("--------------------------------------------------------------------")

    def ajoute(self,tensor):
        self.t0_m += tensor[0]
        self.p0_m += tensor[1]
        self.v0_m += tensor[2]
        self.log_var_xi += tensor[3]
        self.log_var_tau +=tensor[4]
        self.log_var_eps += tensor[5]


class latent:
    def __init__(self,t0,p0,v0,alpha,tau):
        self.t0 = t0
        self.p0= p0
        self.v0 = v0
        #alpha = exp(xi)
        self.alpha = alpha
        self.tau =tau


    def show(self):
        print("t0 :" + str(self.t0) + "   |   p0 :" + str(self.p0) + "   |   v0 :" + str(self.v0) )
        print("liste_alpha : ", self.alpha)
        print("liste_tau : ", self.tau )

    def ajoute(self,tensor,nb_patients):
        self.t0 += tensor[0]
        self.p0+= tensor[1]
        self.v0 += tensor[2]
        for i in range(nb_patients):
            self.alpha[i] += tensor[3+i]
            self.tau[i] +=tensor[3+i+nb_patients]


#lance une géodesique pour la métrique g en un point p0 à la vitesse v0 (renvoie la matrice des gamma(t_ij))
def geodesic(p0,v0,g,t_ij):
    F = lambda t,y:torch.tensor(v0 * g(y)/g(p0))
    
    #le solver prend un tableau trié sans doublons
    #on trie
    t_sort = torch.sort(torch.cat([t_ij[i] for i in range(len(t_ij))]))
    t= torch.tensor(torch.unique(t_sort.values))

    #on enlève les doublons 
    index_t = [0]
    compte = 0
    for i in range(len(t_sort.values)-1):
        if(t_sort.values[i+1] != t_sort.values[i]):
            compte += 1
        index_t.append(compte)

    #resolution
    y0 = torch.tensor([[p0], [5.]])
    t_eval = torch.stack((t,t))

    term = to.ODETerm(F)
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)
    jit_solver = torch.compile(solver)

    sol = jit_solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))

    #réindexation pour obtenir une matrice
    gamma = torch.zeros_like(t_ij)
    n=len(t_ij[0])
    for l in range(len(index_t)):
        k = t_sort.indices[l]
        i = (int)(k/n)
        j= k - n*i
        gamma[i][j] = sol.ys[0][index_t[l]]

    return gamma


def prior(t0,t0m,var_t0,p0,p0m,var_p0,v0,v0m,var_v0,xi,var_xi,tau,var_tau):
    prior = 0
    prior += -((t0-t0m)**2)/(2*var_t0)
    prior += -torch.log(torch.tensor(2*torch.pi*var_t0))/2
    prior += -((p0-p0m)**2)/(2*var_p0)
    prior += -torch.log(torch.tensor(2*torch.pi*var_p0))/2
    prior += -((v0-v0m)**2)/(2*var_v0)
    prior += -torch.log(torch.tensor(2*torch.pi*var_v0))/2
    for xi_i in xi:
        prior += -(xi_i**2)/(2*var_xi)
        prior += -torch.log(torch.tensor(2*torch.pi*var_xi))/2
    for tau_i in tau:
        prior += -(tau_i**2)/(2*var_tau)
        prior += -torch.log(torch.tensor(2*torch.pi*var_tau))/2

    return prior


class langevin:
    def __init__(self,theta_init,var_p0,var_t0,var_v0,nb_patients,nb_mesures):
        self.theta = theta_init
        self.var_p0 = var_p0
        self.var_v0 = var_v0
        self.var_t0 = var_t0
        self.nb_patients = nb_patients
        self.nb_mesures = nb_mesures
        #la metrique est conforme et repesentée par l'inverse de sa racine carrée
        self.metric = lambda x:1
        #future liste des samplings
        self.z_array = []

    def set_metric_logistic(self):
        self.metric = lambda x:x*(1-x)

    def set_metric(self,metric):
        self.metric = metric

    def init_z(self):
        #initialisation du sampling
        p0 = np.random.normal(self.theta.p0_m,np.sqrt(self.var_p0),1)[0]
        t0 = np.random.normal(self.theta.t0_m,np.sqrt(self.var_t0),1)[0]
        v0 = np.random.normal(self.theta.v0_m,np.sqrt(self.var_v0),1)[0]
        alpha = np.exp(np.random.normal(0,np.sqrt(np.exp(self.theta.log_var_xi)),self.nb_patients))
        tau = np.random.normal(0,np.sqrt(np.exp(self.theta.log_var_tau)),self.nb_patients)

        self.z_array.append(latent(p0,t0,v0,alpha,tau))


    #potentiel -> log(p(y,z | theta))
    def potentiel(self,data_t,data_y,t0_m,p0_m,v0_m,var_xi,var_tau,var_eps,t0,p0,v0,xi,tau):
        alpha = torch.exp(xi)

        psi_t = torch.tensor(data_t)
        for i in range(len(data_t)):
            psi_t[i] = alpha[i]*(psi_t[i] - t0 - tau[i])

        gammai = geodesic(p0,v0,self.metric,psi_t)
        S = torch.sum((torch.tensor(data_y) - gammai)**2)

        log_like_ycondz = -1/(2*var_eps) * S - self.nb_patients * len(data_t[0])* torch.log(2*torch.pi*var_eps)/2
        log_like_prior = prior(t0,t0_m,self.var_t0,p0,p0_m,self.var_p0,v0,v0_m,self.var_v0,xi,var_xi,tau,var_tau)

        U = log_like_prior + log_like_ycondz

        return U
        

    #grad_z q(y,z|theta) , grad_theta q(y,z|theta)
    def compute_autograd(self,z,data_t,data_y):

        zlist = [z.t0,z.p0,z.v0]
        for i in range(self.nb_patients):
            zlist.append(np.log(z.alpha[i]))
        for i in range(self.nb_patients):
            zlist.append(z.tau[i])
        

        z_torch = torch.tensor(zlist,requires_grad = True)
        theta_torch = torch.tensor([self.theta.t0_m,self.theta.p0_m,self.theta.v0_m,self.theta.log_var_xi,self.theta.log_var_tau,self.theta.log_var_eps],requires_grad = True)
        xi_torch = z_torch[3:3+self.nb_patients]
        tau_torch = z_torch[3+self.nb_patients:]

        U = self.potentiel(data_t,data_y,theta_torch[0],theta_torch[1],theta_torch[2],torch.exp(theta_torch[3]),torch.exp(theta_torch[4]),torch.exp(theta_torch[5]),z_torch[0],z_torch[1],z_torch[2],xi_torch,tau_torch)

        U.backward()
        return [z_torch.grad,theta_torch.grad]
    
        #le gradient est //t aux xi et non aux alpha !!!
    
    
    def ULA_descente_step(self,var_bruit,stepsize,data_t,data_y):
        m = len(self.z_array)
        noise = [torch.normal(torch.zeros(3+2*self.nb_patients),torch.sqrt(torch.tensor(var_bruit))) for i in range(m)]       

        grad_z_theta =[]
        for i in range(len(self.z_array)):
            grad_z_theta.append(self.compute_autograd(self.z_array[i],data_t,data_y))

        grad_sum = torch.zeros(6)
        for i in range(m):
            grad_sum += grad_z_theta[i][1]

        print("\n",grad_sum)
        
        update_theta = np.sqrt(2*stepsize/m)*torch.normal(torch.zeros(6),var_bruit) - stepsize*grad_sum/m
        self.theta.ajoute(update_theta)


        update_z = [np.sqrt(2*stepsize)*noise[i] - stepsize*grad_z_theta[i][0] for i in range(m)]
        for i in range(m):
            update_xi = update_z[i][3:3+self.nb_patients]
            nouv_alpha = np.exp(np.log(self.z_array[i].alpha) + np.array(update_xi))

            self.z_array[i].ajoute(update_z[i],self.nb_patients)
            self.z_array[i].alpha = nouv_alpha


def ULA_descente(data_y,data_t,theta_init,metric,var_p0,var_t0,var_v0,var_bruit,nb_part,nb_iter,stepsize):
    nb_patients = len(data_t)
    ula = langevin(theta_init,var_p0,var_t0,var_v0,nb_patients,len(data_t[0]))
    print("theta_debut")
    ula.theta.show()
    ula.set_metric(metric)
    for i in range(nb_part):
        ula.init_z()

    for i in range(nb_iter):
        ula.ULA_descente_step(var_bruit,stepsize[i],data_t,data_y)
        ula.theta.show()

    return ula.theta


param = parametres(0.2,0.1,1.,.2,.1,0.4)
t = torch.tensor( [ [1.,2.,3.,4.,5.],[2.,3.,4.,5.,6.]] ) 
y = [[0.12,0.23,0.31,0.38,0.47],[0.32,.39,.51,.58,.71]]
nb_iter = 2000
ULA_descente(y,t,param,lambda x:(x/x),.1,.1,.1,.1,1,nb_iter,[0.0001 for i in range(nb_iter+1)])

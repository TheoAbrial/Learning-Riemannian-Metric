import numpy as np
import torch
import scipy.integrate
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import torchode as to


class parametres:
    def __init__(self,t0_mean,p0_mean,v0_mean,var_xi,var_tau,var_eps):
        self.t0_m = t0_mean
        self.p0_m = p0_mean
        self.v0_m = v0_mean
        self.var_xi = var_xi
        self.var_tau =var_tau
        self.var_eps = var_eps

    def show(self):
        print("t0_mean :" + str(self.t0_m) + "   |   p0_mean :" + str(self.p0_m) + "   |   v0_mean :" + str(self.v0_m) + "\n")
        print("var_xi :" + str(self.var_xi) + "   |   var_tau :" + str(self.var_tau)  + "   |   var_eps :" + str(self.var_eps)  + "\n")
        print("--------------------------------------------------------------------")

    def ajoute(self,tensor):
        self.t0_m += tensor[0]
        self.p0_m += tensor[1]
        self.v0_m += tensor[2]
        self.var_xi += tensor[3]
        self.var_tau +=tensor[4]
        self.var_eps += tensor[5]


class latent:
    def __init__(self,t0,p0,v0,alpha,tau):
        self.t0 = t0
        self.p0= p0
        self.v0 = v0
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

    

def compute_polynomial(roots,x):
    P_x = 1
    for a in roots :
        P_x *= (x - a)
    return P_x


def euler1(F,p0,pas,N,dir):
    p = torch.zeros(N)
    p[0] = p0
    for i in range(N-1):
        p_act = p[i].clone()
        p[i+1] = p_act + dir*pas * F(p_act)
    return p


def geodesic(t0,p0,v0,P, t_ij, N,T):
    F = lambda y:v0 * P(y)/P(p0)
    
    #resoud l'equation pour la geodesique entre t_0 et t_0 + max(t_ij) pour des espacements reguliers
    #print(T)
    #print(t_ij)
    pas = (T-t0)/N  
    y0 = euler1(F, p0,pas,N,1)
    y_rev = euler1(F,p0,pas,N,-1)

    y = torch.cat((torch.flip(y_rev,[0]), y0[1:]))
    #Calcule gamma(t_ij) en interpolant la solution precedente avec des droites 
    gamma = torch.zeros_like(t_ij)
    for i in range(len(gamma)):
        for j in range(len(gamma[0])):
            avancement = (t_ij[i][j]-t0)/pas
            indice_proche = (int)(avancement) + N - 1
            gamma[i][j] = y[indice_proche].clone()
            if(indice_proche != 2*N-2):
                gamma[i][j]= gamma[i][j] + (t_ij[i][j]-t0-indice_proche)*pas*(y[indice_proche+1]-y[indice_proche])
    return gamma

def geodefficace(t0,p0,v0,g,t_ij):
    F = lambda t,y:torch.tensor(v0 * g(y)/g(p0))


    t_sort = torch.sort(torch.cat([t_ij[i] for i in range(len(t_ij))]))
    t= torch.unique(t_sort.values)
    index_t = [0]
    compte = 0
    for i in range(len(t_sort.values)-1):
        if(t_sort.values[i+1] != t_sort.values[i]):
            compte += 1
        index_t.append(compte)

    sol = odeint(F, torch.tensor(p0),t)

    gamma = torch.zeros_like(t_ij)
    n=len(t_ij[0])
    for l in range(len(index_t)):
        print(l)
        k = t_sort.indices[l]
        i = (int)(k/n)
        j= k - n*i
        gamma[i][j] = sol[index_t[l]]

    return gamma

def geodesic(p0,v0,g,t_ij):
    F = lambda t,y:torch.tensor(v0 * g(y)/g(p0))


    t_sort = torch.sort(torch.cat([t_ij[i] for i in range(len(t_ij))]))
    t= torch.tensor(torch.unique(t_sort.values))
    index_t = [0]
    compte = 0
    for i in range(len(t_sort.values)-1):
        if(t_sort.values[i+1] != t_sort.values[i]):
            compte += 1
        index_t.append(compte)

    y0 = torch.tensor([[p0], [p0]])
    t_eval = torch.stack((t,t))

    term = to.ODETerm(F)
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)
    jit_solver = torch.compile(solver)

    sol = jit_solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))

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
        #la metrique est congorme et repesentée par l'inverse de sa racine carrée
        self.metric = lambda x:1
        #future liste des samplings
        self.z_array = []
        self.stats = []
        self.Sk = torch.zeros(9)

    def set_metric_logistic(self):
        self.metric = lambda x:x*(1-x)

    def set_metric(self,metric):
        self.metric = metric

    def update_stats(self):
        #la dernière statistique est mise a jour pendant le calcul du gradient
        stats = []
        for z in self.z_array:
            sum_tau2 = 0
            for tau in z.tau:
                sum_tau2 += tau**2 
                
            sum_xi2 = 0
            for alpha in z.alpha:
                sum_xi2 += np.log(alpha)**2 
        
            stats.append(torch.tensor([-z.t0**2/2,-z.p0**2/2,-z.v0**2/2,z.t0,z.p0,z.v0,-sum_tau2/2,-sum_xi2/2,0]))


        for i in range(len(self.stats)):
           stats[i][8] = self.stats[i][8]

        self.stats = stats


    def init_z(self):
        #initialisation du sampling (choix arbitraire de gaussiennes)
        p0 = np.random.normal(self.theta.p0_m,self.var_p0,1)[0]
        t0 = np.random.normal(self.theta.t0_m,self.var_t0,1)[0]
        v0 = np.random.normal(self.theta.v0_m,self.var_v0,1)[0]
        alpha = np.exp(np.random.normal(0,self.theta.var_xi,self.nb_patients))
        tau = np.random.normal(0,self.theta.var_tau,self.nb_patients)

        self.z_array.append(latent(p0,t0,v0,alpha,tau))
        

    #il faut que j'écrive les caluls dans le latex
    #la fonction fait le calcul du gradient pour une metrique fixee
    def compute_grad(self,z,data_t,data_y):

        #ligne a retirer plus tard (quand on optimisera la metrique)
        self.set_metric_logistic()

        z_torch = torch.tensor([z.p0,z.v0], requires_grad=True)
        P_p0 =self.metric(z_torch[0])

        C0 = z_torch[1]/P_p0
        C0.backward()
        grad_C0 = z_torch.grad
        grad_C1 = torch.tensor([1/P_p0,0])
        
        psi = np.array([lambda t: z.alpha[i]*(t - z.t0 - z.tau[i]) for i in range(self.nb_patients)])
        psi_t = torch.tensor(data_t)
        for i in range(len(data_t)):
            psi_t[i] = psi[i](psi_t[i])
            

        gamma0 = geodesic(z.p0,z.v0,self.metric,psi_t)

        sum_grad = 0
        
        for i in range(self.nb_patients):
            for j in range(len(data_t[0])):
                P_gamma0 = self.metric(gamma0[i][j])

                grad_gamma0= P_gamma0 * (grad_C0*psi_t[i][j] + grad_C1)
                grad_gamma0 = torch.cat((grad_gamma0,torch.zeros(2*self.nb_patients+1)))

                grad_psi = torch.zeros(3+2*self.nb_patients)
                grad_psi[2] = -z.alpha[i]
                grad_psi[3+i] = data_t[i][j]-z.t0 - z.tau[i]
                grad_psi[3+self.nb_patients + i] = -z.alpha[i]

                grad_gammai = grad_gamma0 + C0*P_gamma0*grad_psi

                sum_grad += grad_gammai * (data_y[i][j] - gamma0[i][j])
                
        return sum_grad

    #grad_z q(y,z|theta) , grad_theta q(y,z|theta)
    def compute_autograd(self,z,data_t,data_y):
        self.set_metric_logistic()


        zlist = [z.t0,z.p0,z.v0]
        for i in range(self.nb_patients):
            zlist.append(np.log(z.alpha[i]))
        for i in range(self.nb_patients):
            zlist.append(z.tau[i])
        

        z_torch = torch.tensor(zlist,requires_grad = True)
        theta_torch = torch.tensor([self.theta.t0_m,self.theta.p0_m,self.theta.v0_m,self.theta.var_xi,self.theta.var_tau,self.theta.var_eps],requires_grad = True)
        xi_torch = z_torch[3:3+self.nb_patients]
        alpha_torch =torch.exp(xi_torch)
        tau_torch = z_torch[3+self.nb_patients:]

        psi_t1 = torch.tensor(data_t)
        for i in range(len(data_t)):
            psi_t1[i] = z.alpha[i]*(psi_t1[i] - z.t0 - z.tau[i]) 

        
        T = max(torch.Tensor.item(torch.max(psi_t1)),-torch.Tensor.item(torch.min(psi_t1)))

        psi_t = torch.tensor(data_t)
        for i in range(len(data_t)):
            psi_t[i] = alpha_torch[i]*(psi_t[i] - z_torch[0]- tau_torch[i])
        
         
        #T = max(z.alpha)* (torch.Tensor.item(torch.tensor(max(max(data_t)))) - torch.Tensor.item(torch.tensor(z.t0) - torch.Tensor.item(torch.tensor(min(z.tau)))))
        #T = max(T,-max(z.alpha)* (torch.Tensor.item(torch.tensor(min(min(data_t)))) - torch.Tensor.item(torch.tensor(z.t0) - torch.Tensor.item(torch.tensor(max(z.tau))))))

        gammai = geodesic(z_torch[1],z_torch[2],self.metric,psi_t)
        S = torch.sum((torch.tensor(data_y) - gammai)**2)

        for i in range(len(self.z_array)):
            if(z == self.z_array[i]):
                self.update_stats()
                self.stats[i][8] = -torch.Tensor.item(S)/2

        log_like_ycondz = -1/(2*(theta_torch[5])) * S - self.nb_patients * len(data_t[0])* torch.log(2*torch.pi*theta_torch[5])/2
        log_like_prior = prior(z_torch[0],theta_torch[0],self.var_t0,z_torch[1],theta_torch[1],self.var_p0,z_torch[2],theta_torch[2],self.var_v0,torch.log(alpha_torch),theta_torch[3],tau_torch,theta_torch[4])
        
        somme = log_like_prior + log_like_ycondz
        somme.backward()
        return [z_torch.grad,theta_torch.grad]
    
        #le gradient est //t aux xi et non aux alpha !!!
    
    
    def ULA_descente_step(self,var_walk_noise,stepsize,data_t,data_y):
        m = len(self.z_array)
        noise = [torch.normal(torch.zeros(3+2*self.nb_patients),torch.tensor(var_walk_noise)) for i in range(m)]

        grad_z_theta = [self.compute_autograd(z,data_t,data_y) for z in self.z_array]

        grad_sum = torch.zeros(6)
        for i in range(m):
            grad_sum += grad_z_theta[i][1]

        print("\n",grad_sum)
        
        update_theta = np.sqrt(2*stepsize/m)*torch.normal(torch.zeros(6),var_walk_noise) - stepsize*grad_sum/m


        self.theta.ajoute(update_theta)

        update_z = [np.sqrt(2*stepsize)*noise[i] - stepsize*grad_z_theta[i][0] for i in range(m)]

        for i in range(m):
            update_xi = update_z[i][3:3+self.nb_patients]
            nouv_alpha = np.exp(np.log(self.z_array[i].alpha) + update_xi)

            self.z_array[i].ajoute(update_z[i],self.nb_patients)
            self.z_aray[i].alpha = nouv_alpha

    
    def maximise_theta(self):
        self.theta.t0m = self.Sk[3]
        self.theta.p0m = self.Sk[4]
        self.theta.v0m = self.Sk[5]
        self.theta.var_tau = -2*self.Sk[6]/self.nb_patients
        self.theta.var_xi = -2*self.Sk[7]/self.nb_patients
        self.theta.var_eps= -2*self.Sk[8]/(self.nb_patients*self.nb_mesures)

    def update_z(self,var_walk_noise,stepsize_sampling,data_t,data_y):
        m = len(self.z_array)
        noise = [torch.normal(torch.zeros(3+2*self.nb_patients),torch.tensor(var_walk_noise)) for i in range(m)]
        grad_z = [self.compute_autograd(z,data_t,data_y)[0] for z in self.z_array]
        update_z = [np.sqrt(2*stepsize_sampling)*noise[i] - stepsize_sampling*grad_z[i] for i in range(m)]
        for i in range(m):
            self.z_array[i].ajoute(update_z[i],self.nb_patients)


    def ULA_SAEM_step(self,var_walk_noise,stepsize_sampling,stepsize_S,data_t,data_y):
        
        self.update_z(var_walk_noise,stepsize_sampling,data_t,data_y)
        self.update_stats()

        S_moyen = torch.zeros(9)
        for S in self.stats:
            S_moyen += S
        print("\n S:",S_moyen)
        S_moyen = S_moyen/len(self.stats)
        self.Sk = self.Sk + stepsize_S*(S_moyen - self.Sk)

        self.maximise_theta()    

def ULA_descente(data_y,data_t,theta_init,metric,var_p0,var_t0,var_v0,var_walk_noise,nb_part,nb_iter,stepsize):
    nb_patients = len(data_t)
    ula = langevin(theta_init,var_p0,var_t0,var_v0,nb_patients,len(data_t[0]))
    ula.set_metric(metric)
    for i in range(nb_part):
        ula.init_z()

    for i in range(nb_iter):
        ula.ULA_descente_step(var_walk_noise,stepsize[i],data_t,data_y)
        ula.theta.show()

    return ula.theta

def ULA_SAEM(data_y,data_t,theta_init,metric,var_p0,var_t0,var_v0,var_walk_noise,nb_part,nb_iter,stepsize_sampling,stepsize_S):
    nb_patients = len(data_t)
    ula = langevin(theta_init,var_p0,var_t0,var_v0,nb_patients,len(data_t[0]))
    ula.set_metric(metric)

    for i in range(nb_part):
        ula.init_z()

    ula.update_z(var_walk_noise,stepsize_sampling[i],data_t,data_y)
    ula.Sk = ula.stats[0]

    for i in range(nb_iter):
        ula.ULA_SAEM_step(var_walk_noise,stepsize_sampling[i+1],stepsize_S[i],data_t,data_y)
        ula.z_array[0].show()
        ula.theta.show()

    return ula.theta


param = parametres(0.2,0.1,0.3,.6,.1,10)
t = torch.tensor( [ [1.,2.,3.],[2.,3.,4.]] ) 
y = [[0.12,0.19,0.31],[0.33,.41,.48]]
nb_iter = 2000
ULA_descente(t,y,param,lambda x:1,.1,.1,.1,.1,2,nb_iter,[0.0005 for i in range(nb_iter)])

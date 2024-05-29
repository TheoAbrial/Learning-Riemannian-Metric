import numpy as np
import torch
import scipy.integrate

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


def prior(t0,t0m,var_t0,p0,p0m,var_p0,v0,v0m,var_v0,xi,var_xi,tau,var_tau):
    prior = 0
    prior += -((t0-t0m)**2)/(2*var_t0)
    prior += torch.log(torch.tensor(2*torch.pi*var_t0))/2
    prior += -((p0-p0m)**2)/(2*var_p0)
    prior += torch.log(torch.tensor(2*torch.pi*var_p0))/2
    prior += -((v0-v0m)**2)/(2*var_v0)
    prior += torch.log(torch.tensor(2*torch.pi*var_v0))/2
    for xi_i in xi:
        prior += -(xi_i**2)/(2*var_xi)
        prior += torch.log(torch.tensor(2*torch.pi*var_xi))/2
    for tau_i in tau:
        prior += -(tau_i**2)/(2*var_tau)
        prior += torch.log(torch.tensor(2*torch.pi*var_tau))/2

    return prior


class langevin:
    def __init__(self,theta_init,var_p0,var_t0,var_v0,nb_patients):
        self.theta = theta_init
        self.var_p0 = var_p0
        self.var_v0 = var_v0
        self.var_t0 = var_t0
        self.nb_patients = nb_patients
        #la metrique est congorme et repesentée par l'inverse de sa racine carrée
        self.metric = lambda x:1
        #future liste des samplings
        self.z_array = []

    def set_metric_logistic(self):
        self.metric = lambda x:x*(1-x)

    def set_metric(self,metric):
        self.metric = metric

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
    def compute_grad(self,z,data_t,data_y,N_geo):

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
            

        gamma0 = geodesic(z.t0,z.p0,z.v0,self.metric,psi_t,N_geo)

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
    def compute_autograd(self,z,data_t,data_y,N_geo):
        self.set_metric_logistic()


        zlist = [z.t0,z.p0,z.v0]
        for i in range(self.nb_patients):
            zlist.append(np.log(z.alpha[i]))
        for i in range(self.nb_patients):
            zlist.append(z.tau[i])
        
        z_torch = torch.tensor(zlist,requires_grad = True)
        theta_torch = torch.tensor([self.theta.t0_m,self.theta.p0_m,self.theta.v0_m,self.theta.var_xi,self.theta.var_tau,self.theta.var_eps],requires_grad = True)

        psi_t = torch.tensor(data_t)
        for i in range(len(data_t)):
            psi_t[i] = z_torch[3+i]*(psi_t[i] - z_torch[0]- z_torch[self.nb_patients+3+i])

        gammai = geodesic(z_torch[0],z_torch[1],z_torch[2],self.metric,psi_t,N_geo,100*self.theta.var_xi*max(max(data_t)))
        aux = torch.sum((torch.tensor(data_y) - gammai)**2)
        print("S : ",aux)
        log_like_ycondz = -1/(2*theta_torch[5]) * aux + torch.log(2*torch.pi*z_torch[5])/2
        log_like_prior = prior(z.t0,theta_torch[0],self.var_t0,z.p0,theta_torch[1],self.var_p0,z.v0,theta_torch[2],self.var_v0,np.exp(z.alpha),theta_torch[3],z.tau,theta_torch[4])
        
        somme = log_like_prior + log_like_ycondz
        somme.backward()
        return [z_torch.grad,theta_torch.grad]
    
    def ULA_step(self,var_walk_noise,stepsize,data_t,data_y,N_geo):
        m = len(self.z_array)
        noise = [torch.normal(torch.zeros(3+2*self.nb_patients),torch.tensor(var_walk_noise)) for i in range(m)]

        grad_z_theta = [self.compute_autograd(z,data_t,data_y,N_geo) for z in self.z_array]

        grad_sum = torch.zeros(6)
        for i in range(m):
            grad_sum += grad_z_theta[i][1]

        
        update_theta = np.sqrt(2*stepsize/m)*torch.normal(torch.zeros(6),var_walk_noise) - grad_sum

        self.theta.ajoute(update_theta)

        update_z = [np.sqrt(2*stepsize)*noise[i] - stepsize*grad_z_theta[i][0] for i in range(m)]

        for i in range(m):
            self.z_array[i].ajoute(update_z[i],self.nb_patients)



def ULA(data_y,data_t,theta_init,metric,var_p0,var_t0,var_v0,var_walk_noise,nb_part,N_geo,nb_iter,stepsize):
    nb_patients = len(data_t)
    ula = langevin(theta_init,var_p0,var_t0,var_v0,nb_patients)
    ula.set_metric(metric)
    for i in range(nb_part):
        ula.init_z()

    for i in range(nb_iter):
        ula.ULA_step(var_walk_noise,stepsize[i],data_t,data_y,N_geo)
        ula.theta.show()

    return ula.theta


param = parametres(0.5,0.5,0.5,.6,.1,10)
t = [[1.,2.,3.],[2.,3.,4.],[4.,5.,6.]]
y = [[0.1,0.2,0.3],[0.3,.4,.5],[0.3,0.38,.45]]
ULA(t,y,param,lambda x:x*(1-x),.1,.1,.1,.1,1,100,10,[.001 for i in range(20)])

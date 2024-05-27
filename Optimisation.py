import numpy as np
import torch
import scipy.integrate

class parametres:
    def __init__(self,t0_mean,p0_mean,v0_mean,sigma_xi,sigma_tau,sigma_eps):
        self.t0_mean = t0_mean
        self.p0_mean = p0_mean
        self.v0_mean = v0_mean
        self.sigma_xi = sigma_xi
        self.sigma_tau =sigma_tau
        self.sigma_eps = sigma_eps

    def show(self):
        print("t0_mean :" + str(self.t0_mean) + "   |   p0_mean :" + str(self.p0_mean) + "   |   v0_mean :" + str(self.v0_mean) + "\n")
        print("sigma_xi :" + str(self.sigma_xi) + "   |   sigma_tau :" + str(self.sigma_tau)  + "   |   sigma_eps :" + str(self.sigma_eps)  + "\n")


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
    

def compute_polynomial(roots,x):
    P_x = 1
    for a in roots :
        P_x *= (x - a)
    return P_x


def geodesic(t_0,p_0, v_0,P, t_ij, N):

    #Renvoie gamma_0(t_ij) ou t_ij est un tab de temps en entrée et g=1/P² est la metrique
    F = lambda y,t : v_0*P(y)/P(p_0)

    #resoud l'equation pour la geodesique entre t_0 et t_0 + max(t_ij) pour des espacements reguliers
    T = torch.max(t_ij)
    T = T +0.001
    t = np.linspace(t_0, t_0+T, N)
    pas = (T-t_0)/N  
    y0 = scipy.integrate.odeint(F, p_0, t)
    y_rev = scipy.integrate.odeint(F, p_0, -t+2*t_0)
    y = np.concatenate((np.concatenate(np.flip(y_rev)), np.delete(y0,[0])))
    print(y)
    #Calcule gamma(t_ij) en interpolant la solution precedente avec des droites 
    gamma = np.zeros_like(t_ij)
    for i in range(len(gamma)):
        for j in range(len(gamma[0])):
            avancement = (t_ij[i][j]-t_0)/pas
            indice_proche = (int)(avancement) + N - 1
            gamma[i][j] = y[indice_proche]
            if(indice_proche != 2*N-2):
                gamma[i][j]+= (t_ij[i][j]-t_0-indice_proche)*pas*(y[indice_proche+1]-y[indice_proche])
                
    print(gamma)
    return gamma


class langevin:
    def __init__(self,theta_init,sigma_p0,sigma_t0,sigma_v0,nb_patients):
        self.theta = theta_init
        self.sigma_p0 = sigma_p0
        self.sigma_v0 = sigma_v0
        self.sigma_t0 = sigma_t0
        self.nb_patients = nb_patients
        #la metrique sera de la forme 1/P^2 où P est un polynome unitaire qu'on representera par la liste de ses racines (en dehors de ]0,1[ )
        self.metric = []
        #future liste des samplings
        self.z_array = []

    def set_metric_logistic(self):
        self.metric = np.array([0,1])

    def init_z(self):
        #initialisation du sampling (choix arbitraire de gaussiennes)
        p0 = np.random.normal(self.theta.p0_mean,self.sigma_p0,1)[0]
        t0 = np.random.normal(self.theta.t0_mean,self.sigma_t0,1)[0]
        v0 = np.random.normal(self.theta.v0_mean,self.sigma_v0,1)[0]
        alpha = np.exp(np.random.normal(0,self.theta.sigma_xi,self.nb_patients))
        tau = np.random.normal(0,self.theta.sigma_tau,self.nb_patients)

        self.z_array.append(latent(p0,t0,v0,alpha,tau))
        

    #il faut que j'écrive les caluls dans le latex
    #la fonction fait le calcul du gradient pour une metrique fixee
    def compute_grad(self,z,data_t,data_y,N_geo):

        #ligne a retirer plus tard (quand on optimisera la metrique)
        self.set_metric_logistic()
        P = lambda x: np.prod(np.array([(x-a) for a in self.metric]))

        z_torch = torch.tensor(np.array([z.p0,z.v0]), requires_grad=True)

        P_p0 = 1
        for a in self.metric:
            P_p0 *= z_torch[0]-a

        C0 = z_torch[1]/P_p0
        C0.backward()
        grad_C0 = z_torch.grad
        grad_C1 = torch.tensor([1/P_p0,0])
        
        psi = np.array([lambda t: z.alpha[i]*(t - z.t0 - z.tau[i]) for i in range(self.nb_patients)])
        psi_t = torch.tensor(data_t)
        for i in range(len(data_t)):
            psi_t[i] = psi[i](psi_t[i])
            

        gamma0 = geodesic(z.t0,z.p0,z.v0,P,psi_t,N_geo)

        sum_grad = 0
        
        for i in range(self.nb_patients):
            for j in range(len(data_t[0])):
                P_gamma0 = P(gamma0[i][j])

                grad_gamma0= P_gamma0 * (grad_C0*psi_t[i][j] + grad_C1)
                grad_gamma0 = torch.cat((grad_gamma0,torch.zeros(2*self.nb_patients+1)))

                grad_psi = torch.zeros(3+2*self.nb_patients)
                grad_psi[2] = -z.alpha[i]
                grad_psi[3+i] = data_t[i][j]-z.t0 - z.tau[i]
                grad_psi[3+self.nb_patients + i] = -z.alpha[i]

                grad_gammai = grad_gamma0 + C0*P_gamma0*grad_psi

                sum_grad += grad_gammai * (data_y[i][j] - gamma0[i][j])
                
        return sum_grad
        




param = parametres(0.5,0.5,0.5,0.1,.1,.1)
test = langevin(param,.1,.1,.1,3)
test.init_z()
test.z_array[0].show()
t = [[1.,2.,3.],[2.,3.,4.],[4.,5.,6.]]
y = [[0.1,0.2,0.3],[0.3,.4,.5],[0.3,0.38,.45]]
test.compute_grad(test.z_array[0],t,y,50)

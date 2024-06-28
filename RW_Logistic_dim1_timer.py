import numpy as np
import torch
import matplotlib.pyplot as plt
import random as rd
from timeit import default_timer

def g(p):
    return 1/(p*p*(1-p)*(1-p))

def geodesic(t, p, v):
    return lambda x: 1/(1 + (1/p - 1)*torch.exp((v/(p*(1-p)))*(t - x)) )

def list_sum(l1, l2):
    return [l1[i]+l2[i] for i in range(len(l1))]

def list_mul(l, x):
    return [x*l[i] for i in range(len(l))]


class parametres:
    def __init__(self,t0_mean,p0_mean,v0_mean,logvar_xi,logvar_tau,logvar_eps):
        self.t0_m = t0_mean.clone().detach()
        self.p0_m = p0_mean.clone().detach()
        self.v0_m = v0_mean.clone().detach()
        self.logvar_xi = logvar_xi.clone().detach()
        self.logvar_tau = logvar_tau.clone().detach()
        self.logvar_eps = logvar_eps.clone().detach()

    def show(self):
        print("t0_mean :" + str(self.t0_m) + "   |   p0_mean :" + str(self.p0_m) + "   |   v0_mean :" + str(self.v0_m) + "\n")
        print("logvar_xi :" + str(self.logvar_xi) + "   |   logvar_tau :" + str(self.logvar_tau)  + "   |   logvar_eps :" + str(self.logvar_eps)  + "\n")
        print("--------------------------------------------------------------------")

    def ajoute(self,tensor):
        self.t0_m += tensor[0]
        self.p0_m += tensor[1]
        self.v0_m += tensor[2]
        self.logvar_xi += tensor[3]
        self.logvar_tau +=tensor[4]
        self.logvar_eps += tensor[5]
    


class latent:
    def __init__(self,t0,p0,v0,xi,tau):
        self.t0 = t0
        self.p0= p0
        self.v0 = v0
        self.xi = xi
        self.tau = tau


    def show(self):
        print("t0 :" + str(self.t0) + "   |   p0 :" + str(self.p0) + "   |   v0 :" + str(self.v0) )
        #print("liste_xi : ", self.xi)
        #print("liste_tau : ", self.tau )

    def ajoute(self,tensor,nb_patients):
        self.t0 += tensor[0]
        self.p0+= tensor[1]
        self.v0 += tensor[2]
        for i in range(nb_patients):
            self.xi[i] += tensor[3+i]
            self.tau[i] +=tensor[3+i+nb_patients]

    def tensorize(self):
        L = [self.t0, self.p0, self.v0]
        for elem in self.xi:
            L.append(elem)
        for elem in self.tau :
            L.append(elem)
        return L

    def add_latent(self, z):
        nb_patients = len(self.xi)

        self.t0 += z.t0
        self.p0 += z.p0
        self.v0 += z.v0
        for i in range(nb_patients):
            self.xi[i] += z.xi[i]
            self.tau[i] += z.tau[i]

    def mul(self, x):
        nb_patients = len(self.xi)
        
        self.t0 = self.t0 * x
        self.p0 = self.p0 * x
        self.v0 = self.v0 * x
        for i in range(nb_patients):
            self.xi[i] = self.xi[i] * x
            self.tau[i] = self.tau[i] * x

    def norm(self):
        s = 0
        s += self.t0**2
        s += self.p0**2
        s += self.v0**2
        for i in range(len(self.xi)):
            s += self.xi[i]**2
            s += self.tau[i]**2
        return s

    def set(self, z):
        nb_patients = len(self.xi)
        
        self.t0 = z.t0
        self.p0 = z.p0
        self.v0 = z.v0
        for i in range(nb_patients):
            self.xi[i] = z.xi[i]
            self.tau[i] = z.tau[i]
    
    def clone(self):
        z_t0 = self.t0.clone().detach()
        z_p0 = self.p0.clone().detach()
        z_v0 = self.v0.clone().detach()
        z_xi = []
        z_tau = []

        for i in range(len(self.xi)):
            z_xi.append(self.xi[i].clone().detach())
            z_tau.append(self.tau[i].clone().detach())

        z = latent(z_t0, z_p0, z_v0, z_xi, z_tau)
        return z

def randomize(n, var_t0, var_p0, var_v0, logvar_xi, logvar_tau):
    z_t0 = rd.gauss(0, torch.sqrt(torch.tensor(var_t0)))
    z_p0 = rd.gauss(0, torch.sqrt(torch.tensor(var_p0)))
    z_v0 = rd.gauss(0, torch.sqrt(torch.tensor(var_v0)))
    z_xi = [rd.gauss(0, torch.exp(torch.tensor(logvar_xi)/2)) for i in range(n)]
    z_tau = [rd.gauss(0, torch.exp(torch.tensor(logvar_tau)/2)) for i in range(n)]
    return latent(z_t0, z_p0, z_v0, z_xi, z_tau)


def prior(t0,t0m,var_t0,p0,p0m,var_p0,v0,v0m,var_v0,xi,logvar_xi,tau,logvar_tau):
    prior = 0
    prior += -((t0-t0m)**2)/(2*var_t0)
    prior += -torch.log(2*torch.pi*var_t0)/2
    prior += -((p0-p0m)**2)/(2*var_p0)
    prior += -torch.log(2*torch.pi*var_p0)/2
    prior += -((v0-v0m)**2)/(2*var_v0)
    prior += -torch.log(2*torch.pi*var_v0)/2
    for xi_i in xi:
        prior += -(xi_i**2)/(2*torch.exp(logvar_xi))
        prior += -torch.log(torch.tensor(2*torch.pi))/2 -logvar_xi/2
    for tau_i in tau:
        prior += -(tau_i**2)/(2*torch.exp(logvar_tau))
        prior += -torch.log(torch.tensor(2*torch.pi))/2 - logvar_tau/2

    return prior

def epsilon(Nb, beta, k):
    if k > Nb:
        return pow(k-Nb, -1 * beta)
    else:
        return 1

def indic(b):
    if b:
        return 1
    else:
        return 0
        
        

class MCMC:
    def __init__(self,data_t, data_y, theta_init, var_t0, var_p0, var_v0, nb_patients, nb_mesures):
        self.theta = theta_init
        self.var_p0 = var_p0.clone().detach()
        self.var_v0 = var_v0.clone().detach()
        self.var_t0 = var_t0.clone().detach()
        self.nb_patients = nb_patients
        self.nb_mesures = nb_mesures
        self.data_t = data_t
        self.data_y = data_y
        self.iter = 0
        self.z_array = []
        self.stats = []
        self.err = torch.tensor(0.)
        self.Sk = torch.zeros(9)

    def init_z(self):
        #initialisation du sampling (choix arbitraire de gaussiennes)
        t0 = rd.gauss(self.theta.t0_m,torch.sqrt(self.var_t0))
        p0 = rd.gauss(self.theta.p0_m,torch.sqrt(self.var_p0))
        v0 = rd.gauss(self.theta.v0_m,torch.sqrt(self.var_v0))
        xi = [rd.gauss(0, torch.exp(self.theta.logvar_xi/2)) for i in range(self.nb_patients)]
        tau = [rd.gauss(0, torch.exp(self.theta.logvar_tau/2)) for i in range(self.nb_patients)]

        self.z_array.append(latent(p0,t0,v0,xi,tau))
        self.err = self.compute_err(t0, p0, v0, xi, tau)

    def compute_err(self, z_t0, z_p0, z_v0, z_xi, z_tau):
        # Calcule sum_{i,j} (yij - eta^w_i(gamma_0, psi_i(tij))**2
        geo = geodesic(z_t0, z_p0, z_v0)
        res = 0
        for i in range(self.nb_patients):
            alpha_i = torch.exp(z_xi[i])
            for j in range(len(self.data_t[i])):
                t_ij = (alpha_i)*(self.data_t[i][j] - z_t0 - z_tau[i]) + z_t0
                res  = res + (self.data_y[i][j] - geo(t_ij))**2
        return res

    def update_z(self, eps):
        z_prev = self.z_array[-1]

        zeta_p0 = torch.tensor(0.01**2)
        zeta_v0 = torch.tensor(0.03*2)
        zeta_t0 = torch.tensor(0.01**2)
        zeta_ksi = 2*torch.log(torch.tensor(0.1))
        zeta_tau = 2*torch.log(torch.tensor(0.05))

        r = randomize(self.nb_patients, zeta_t0, zeta_p0, zeta_v0, zeta_ksi, zeta_tau)

        z = z_prev.clone()
        z.add_latent(r)

        self.z_array.append(z)
        self.err = self.compute_err(z.t0, z.p0, z.v0, z.xi, z.tau)
        

    def maximise_theta(self):
        p0_mean = self.Sk[3].clone().detach()
        t0_mean = self.Sk[4].clone().detach()
        v0_mean = self.Sk[5].clone().detach()
        logvar_tau = torch.log(-2*self.Sk[6]/self.nb_patients).clone().detach()
        logvar_xi = torch.log(-2*self.Sk[7]/self.nb_patients).clone().detach()
        logvar_eps = torch.log(-2*self.Sk[8]/(self.nb_mesures)).clone().detach()
        self.theta = parametres(t0_mean.clone().detach(), p0_mean.clone().detach(), v0_mean.clone().detach(), logvar_xi.clone().detach(), logvar_tau.clone().detach(), logvar_eps.clone().detach())

    def compute_S(self, z):
        sum_tau2 = 0
        sum_xi2 = 0
        for i in range(self.nb_patients):
            sum_tau2 += z.tau[i]**2
            sum_xi2 += z.xi[i]**2
        return [-z.p0**2/2, -z.t0**2/2, -z.v0**2/2, z.p0, z.t0, z.v0, -sum_tau2/2, -sum_xi2/2, -self.err/2]

    def update_Sk(self, Nb, beta):
        S_aux = self.compute_S(self.z_array[-1])
        eps = epsilon(Nb, beta, len(self.z_array))
        self.Sk = list_sum(list_mul(self.Sk, 1 - eps), list_mul(S_aux, eps))

        

def RW(data_t, data_y, theta_init, var_t0, var_p0, var_v0, nb_patients, nb_mesures, eps, N, Nb, beta):
    rw = MCMC(data_t, data_y, theta_init, var_t0, var_p0, var_v0, nb_patients, nb_mesures)
    init_timer = default_timer()
    liste_theta = [theta_init]
    liste_timer = [0.]
    rw.init_z()
    for i in range(N):
        print("itération : " +str(i + 1))
        rw.theta.show()
        rw.update_z(eps)
        rw.update_Sk(Nb, beta)
        rw.maximise_theta()
        liste_timer.append(default_timer() - init_timer)
        liste_theta.append(parametres(rw.theta.t0_m, rw.theta.p0_m, rw.theta.v0_m, rw.theta.logvar_xi, rw.theta.logvar_tau, rw.theta.logvar_eps))
    print("ooooooooooooooooooooooooooooooooooooooooooooooo")
    print("")
    print("parametres obtenus :")
    print("")
    rw.theta.show()
    print("")
    print("ooooooooooooooooooooooooooooooooooooooooooooooo")

    '''
    for i in range(N):
        if ((i+1)%25 == 0):
            nuts.z_array[i].show()
    '''
    
    return rw, liste_theta, liste_timer
    

def RW_light(data_t, data_y, theta_init, step_size, N, Nb):

    nb_patients = len(data_t)
    nb_mesures = 0
    for i in range(nb_patients):
        nb_mesures += len(data_t[i])
    
    beta = 0.65

    var_t0 = torch.tensor(0.05**2)
    var_p0 = torch.tensor(0.1**2)
    var_v0 = torch.tensor(0.025**2)

    return RW(data_t, data_y, theta_init, var_t0, var_p0, var_v0, nb_patients, nb_mesures, step_size, N, Nb, beta)

def generate_data(data_t, th, var_t0, var_p0, var_v0):
    p0 = rd.gauss(th.p0_m, torch.sqrt(var_p0))
    v0 = rd.gauss(th.v0_m, torch.sqrt(var_v0))
    t0 = rd.gauss(th.t0_m, torch.sqrt(var_t0))
    geo = geodesic(t0, p0, v0)
    y = []
    for i in range(len(data_t)):
        xi_i = rd.gauss(0, torch.exp(th.logvar_xi/2))
        tau_i = rd.gauss(0, torch.exp(th.logvar_tau/2))
        alpha_i = torch.exp(xi_i)
        y_i = []
        for j in range(len(data_t[i])):
            eps_ij = rd.gauss(0, torch.exp(th.logvar_eps/2))
            t_ij = alpha_i*(data_t[i][j] - t0 - tau_i) + t0
            y_ij = geo(t_ij) + eps_ij
            y_i.append(y_ij)
        y.append(y_i)
    return y, geo

def plot_data(data_t, data_y):
    for i in range(len(data_t)):
        t = np.array([data_t[i][j] for j in range(len(data_t[i]))])
        y = np.array([data_y[i][j] for j in range(len(data_y[i]))])
        plt.plot(t, y, '-')

def plot_geo(times, geo):
    data_time = np.array([times[i] for i in range(len(times))])
    y = np.array([geo(times[i]) for i in range(len(times))])
    plt.plot(data_time, y, '-', linewidth = 3, color = 'black')


########## hyper-paramètres 

var_t0 = torch.tensor(0.05**2)
var_p0 = torch.tensor(0.1**2)
var_v0 = torch.tensor(0.025**2)

########## Nb patients et mesures

nb_patients = 300
nb_mesures = 8 * nb_patients

########## paramètres initiaux

sigma_xi = torch.tensor(0.5)
sigma_tau = torch.tensor(0.1)
sigma_eps = torch.tensor(0.005)

sigma_xi_init = torch.tensor(0.67)
sigma_tau_init = torch.tensor(0.042)
sigma_eps_init = torch.tensor(0.5)

t0_m = torch.tensor(0.5)
p0_m = torch.tensor(0.5)
v0_m = torch.tensor(3)

t0_m_init = torch.tensor(0.43)
p0_m_init = torch.tensor(0.4)
v0_m_init = torch.tensor(2.4)

theta = parametres(t0_m, p0_m, v0_m, 2*torch.log(sigma_xi), 2*torch.log(sigma_tau), 2*torch.log(sigma_eps))
theta_init = parametres(t0_m_init, p0_m_init, v0_m_init, 2*torch.log(sigma_xi_init), 2*torch.log(sigma_tau_init), 2*torch.log(sigma_eps_init))

########## Pour affichage

step_t = 1/((nb_mesures//nb_patients) - 1)
data_t = [[i*step_t for i in range(nb_mesures//nb_patients)] for j in range(nb_patients)]

data_y, geo = generate_data(data_t, theta, var_t0, var_p0, var_v0)

########## Paramètres algorithme

eps = 0.002 # step_size
N = 2000 # nb d'itérations
Nb = 500 # à partir de quand on prend en compte les z dans Sk
beta = 0.65 # la puissance de la suite epsilon_k

rw, liste_theta, liste_timer = RW(data_t, data_y, theta_init, var_t0, var_p0, var_v0, nb_patients, nb_mesures, eps, N, Nb, beta)
theta_rw = rw.theta

liste_erreur_t0 = np.array([[np.abs(liste_theta[i].t0_m - theta.t0_m)] for i in range(len(liste_theta))])
liste_erreur_p0 = np.array([[np.abs(liste_theta[i].p0_m - theta.p0_m)] for i in range(len(liste_theta))])
liste_erreur_v0 = np.array([[np.abs(liste_theta[i].v0_m - theta.v0_m)] for i in range(len(liste_theta))])
liste_erreur_logvar_xi = np.array([[np.abs(liste_theta[i].logvar_xi - theta.logvar_xi)] for i in range(len(liste_theta))])
liste_erreur_logvar_tau = np.array([[np.abs(liste_theta[i].logvar_tau - theta.logvar_tau)] for i in range(len(liste_theta))])
liste_erreur_logvar_eps = np.array([[liste_theta[i].logvar_eps] for i in range(len(liste_theta))])

liste_temps = np.array(liste_timer)

########## Pour affichage

data_y_rw, geo_rw = generate_data(data_t, theta_rw, var_t0, var_p0, var_v0)

plt.figure(1)

plot_data(data_t, data_y)
plot_geo(data_t[0], geo)

plt.figure(2)

plot_data(data_t, data_y_rw)
plot_geo(data_t[0], geo_rw)


plt.figure(3)

plt.title("Erreur t_0 moyen")
plt.xlabel("Temps pris (secondes)")
plt.ylabel("Erreur t_0 moyen")

plt.plot(liste_temps, liste_erreur_t0, '-', color = "black")

plt.figure(4)

plt.title("Erreur p_0 moyen")
plt.xlabel("Temps pris (secondes)")
plt.ylabel("Erreur p_0 moyen")

plt.plot(liste_temps, liste_erreur_p0, '-', color = "black")

plt.figure(5)

plt.title("Erreur v_0 moyen")
plt.xlabel("Temps pris (secondes)")
plt.ylabel("Erreur v_0 moyen")

plt.plot(liste_temps, liste_erreur_v0, '-', color = "black")

plt.figure(6)

plt.title("Erreur log(sigma_xi)")
plt.xlabel("Temps pris (secondes)")
plt.ylabel("Erreur log(sigma_xi)")

plt.plot(liste_temps, liste_erreur_logvar_xi, '-', color = "black")

plt.figure(7)

plt.title("Erreur log(sigma_tau)")
plt.xlabel("Temps pris (secondes)")
plt.ylabel("Erreur log(sigma_tau)")

plt.plot(liste_temps, liste_erreur_logvar_tau, '-', color = "black")

plt.figure(8)

plt.title("logarithme du terme d'erreur (bruit du modèle)")
plt.xlabel("Temps pris (secondes)")

plt.plot(liste_temps, liste_erreur_logvar_eps, '-', color = "green")

plt.show()




















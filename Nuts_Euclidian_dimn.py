import numpy as np
import torch
import matplotlib.pyplot as plt
import random as rd
from CSV_Reader import *
from timeit import default_timer

global d 
d = 2

def geodesic(t, p, v):
    return lambda x: [v[i]*(x - t) + p[i] for i in range(d)] 

def exp_para(geo, w):
    def aux(t):
        geo_t = geo(t)
        return [geo_t[i] + w[i] for i in range(d)]
    return aux

def house_holder(x):
    n = len(x)
    sign = x[0]/torch.abs(x[0])
    xbis = [-1*sign*x[i].clone().detach() for i in range(n)]
    b = []
    for k in range(1, n):
        b_k = []
        b_k.append(xbis[k])
        for l in range(1, n):
            if l == k :
                b_k.append(1 - (xbis[k]*xbis[k])/(1 - xbis[0]))
            else:
                b_k.append(-1 * (xbis[l]*xbis[k])/(1 - xbis[0]))
        b.append(b_k)
    return b

def normalize_temp(data_t):
    # renvoie un nouveau tableau data_t pour que tous les éléments soient entre 
    # "start" et "end" en fonction du plus petit temps et du plus grand temps

    min = data_t[0][0]
    max = data_t[0][0]

    start = 0.1
    end = 0.9

    for i in range(len(data_t)):
        for j in range(len(data_t[i])):
            if data_t[i][j] > max:
                max = data_t[i][j]
            if data_t[i][j] < min:
                min = data_t[i][j]
    diff = (max - min)/(end - start)
    new = []
    for i in range(len(data_t)):
        new_i = []
        for j in range(len(data_t[i])):
            new_i.append(start + (data_t[i][j] - min)/diff)
        new.append(new_i)
    
    return new


def epsilon(Nb, power, k):
    if k > Nb:
        return pow(k-Nb, -1 * power)
    else:
        return 1

def list_sum(l1, l2):
    return [l1[i]+l2[i] for i in range(len(l1))]

def list_mul(l, x):
    return [x*l[i] for i in range(len(l))]


class parametres:
    def __init__(self, t0_mean, p0_mean, v0_mean, beta_mean, logvar_xi, logvar_tau, logvar_eps):
        self.t0_m = t0_mean.clone().detach()
        self.p0_m = [p0_mean[i].clone().detach() for i in range(d)]
        self.v0_m = [v0_mean[i].clone().detach() for i in range(d)]
        self.beta_m = [[beta_mean[l][k].clone().detach() for k in range(d - 1)] for l in range(len(beta_mean))]
        self.Ns = len(beta_mean)
        self.logvar_xi = logvar_xi.clone().detach()
        self.logvar_tau = logvar_tau.clone().detach()
        self.logvar_eps = logvar_eps.clone().detach()

    def show(self):
        print("t0_mean :" + str(self.t0_m) + "   |   p0_mean :" + str(self.p0_m) + "   |   v0_mean :" + str(self.v0_m) + "\n")
        print("logvar_xi :" + str(self.logvar_xi) + "   |   logvar_tau :" + str(self.logvar_tau)  + "   |   logvar_eps :" + str(self.logvar_eps)  + "\n")
        print("--------------------------------------------------------------------")


class latent:
    def __init__(self,t0,p0,v0,xi,tau,s,beta):
        self.t0 = t0
        self.p0 = p0
        self.v0 = v0
        self.s = s
        self.beta = beta
        self.xi = xi
        self.tau = tau
        self.Ns = len(s[0])


    def show(self):
        print("t0 :" + str(self.t0) + "   |   p0 :" + str(self.p0) + "   |   v0 :" + str(self.v0) )
        #print("liste_xi : ", self.xi)
        #print("liste_tau : ", self.tau )


    def add_latent(self, z):
        nb_patients = len(self.xi)

        self.t0 += z.t0
        for i in range(d):
            self.p0[i] += z.p0[i]
            self.v0[i] += z.v0[i]
        for i in range(nb_patients):
            self.xi[i] += z.xi[i]
            self.tau[i] += z.tau[i]
            for l in range(self.Ns):
                self.s[i][l] += z.s[i][l]
        for l in range(self.Ns):
            for k in range(d - 1):
                self.beta[l][k] += z.beta[l][k]

    def mul(self, x):
        nb_patients = len(self.xi)
        
        self.t0 = self.t0 * x
        for i in range(d):
            self.p0[i] = self.p0[i] * x
            self.v0[i] = self.v0[i] * x
        for i in range(nb_patients):
            self.xi[i] = self.xi[i] * x
            self.tau[i] = self.tau[i] * x
            for l in range(self.Ns):
                self.s[i][l] = self.s[i][l] * x
        for l in range(self.Ns):
            for k in range(d - 1):
                self.beta[l][k] = self.beta[l][k] * x

    def norm(self):
        nb_patients = len(self.xi)
        S = 0
        S += self.t0**2
        for i in range(d):
            S += self.p0[i]**2
            S += self.v0[i]**2
        for i in range(nb_patients):
            S += self.xi[i]**2
            S += self.tau[i]**2
            for l in range(self.Ns):
                S += self.s[i][l]**2
        for l in range(self.Ns):
            for k in range(d - 1):
                S += self.beta[l][k]**2
        return S

    def set(self, z):
        nb_patients = len(self.xi)
        
        self.t0 = z.t0
        for i in range(d):
            self.p0[i] = z.p0[i]
            self.v0[i] = z.v0[i]
        for i in range(nb_patients):
            self.xi[i] = z.xi[i]
            self.tau[i] = z.tau[i]
            for l in range(self.Ns):
                self.s[i][l] = z.s[i][l]
        for l in range(self.Ns):
            for k in range(d - 1):
                self.beta[l][k] = z.beta[l][k]
    
    def clone(self):
        z_t0 = self.t0.clone().detach()
        z_p0 = [self.p0[i].clone().detach() for i in range(d)]
        z_v0 = [self.v0[i].clone().detach() for i in range(d)]
        z_xi = []
        z_tau = []
        z_s = []
        z_beta = []

        for i in range(len(self.xi)):
            z_xi.append(self.xi[i].clone().detach())
            z_tau.append(self.tau[i].clone().detach())
            z_s_i = []
            for l in range(self.Ns):
                z_s_i.append(self.s[i][l].clone().detach())
            z_s.append(z_s_i)
        for l in range(self.Ns):
            z_beta_l = []
            for k in range(d - 1):
                z_beta_l.append(self.beta[l][k].clone().detach())
            z_beta.append(z_beta_l)


        z = latent(z_t0, z_p0, z_v0, z_xi, z_tau, z_s, z_beta)
        return z

def randomize(n, Ns, var_t0, var_p0, var_v0, logvar_xi, logvar_tau, var_s, var_beta):
    z_t0 = rd.gauss(0, torch.sqrt(torch.tensor(var_t0)))
    z_p0 = [rd.gauss(0, torch.sqrt(torch.tensor(var_p0))) for i in range(d)]
    z_v0 = [rd.gauss(0, torch.sqrt(torch.tensor(var_v0))) for i in range(d)]
    z_xi = [rd.gauss(0, torch.exp(torch.tensor(logvar_xi)/2)) for i in range(n)]
    z_tau = [rd.gauss(0, torch.exp(torch.tensor(logvar_tau)/2)) for i in range(n)]
    z_s = [[rd.gauss(0, torch.sqrt(torch.tensor(var_s))) for l in range(Ns)] for i in range(n)]
    z_beta = [[rd.gauss(0, torch.sqrt(torch.tensor(var_beta))) for k in range(d - 1)] for l in range(Ns)]
    return latent(z_t0, z_p0, z_v0, z_xi, z_tau, z_s, z_beta)


def prior(t0,t0m,var_t0,p0,p0m,var_p0,v0,v0m,var_v0,xi,logvar_xi,tau,logvar_tau, s, beta, betam, var_beta):
    S = 0
    S -= ((t0-t0m)**2)/(2*var_t0)
    S -= torch.log(var_t0)/2
    for i in range(d):
        S -= ((p0[i]-p0m[i])**2)/(2*var_p0)
        S -= ((v0[i]-v0m[i])**2)/(2*var_v0)
    S -= torch.log(var_p0)/2
    S -= torch.log(var_v0)/2
    for xi_i in xi:
        S -= (xi_i**2)/(2*torch.exp(logvar_xi))
        S -= logvar_xi/2
    for tau_i in tau:
        S -= (tau_i**2)/(2*torch.exp(logvar_tau))
        S -= logvar_tau/2
    for i in range(len(s)):
        for l in range(len(s[i])):
            S -= (s[i][l])**2/2
    for l in range(len(beta)):
        for k in range(len(beta[l])):
            S -= (beta[l][k] - betam[l][k])**2/(2*var_beta)
            S -= torch.log(var_beta)/2

    return S

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
    def __init__(self,data_t, data_y, theta_init,var_t0,var_p0,var_v0,var_beta,nb_patients,nb_mesures,limit_j):
        self.theta = theta_init
        self.limit_j = limit_j
        self.var_p0 = var_p0.clone().detach()
        self.var_v0 = var_v0.clone().detach()
        self.var_t0 = var_t0.clone().detach()
        self.var_beta = var_beta.clone().detach()
        self.nb_patients = nb_patients
        self.nb_mesures = nb_mesures
        self.data_t = data_t
        self.data_y = data_y
        #future liste des samplings
        self.iter = 0
        self.z_array = []
        self.stats = []
        self.err = torch.tensor(0.)
        self.Ns = self.theta.Ns
        self.Sk = [torch.tensor(0.), torch.zeros(d), torch.zeros(d), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), [torch.zeros(d - 1) for l in range(self.Ns)]]
        
    def init_z(self):
        #initialisation du sampling (choix arbitraire de gaussiennes)
        t0 = rd.gauss(self.theta.t0_m,torch.sqrt(self.var_t0))
        p0 = [rd.gauss(self.theta.p0_m[i],torch.sqrt(self.var_p0)) for i in range(d)]
        v0 = [rd.gauss(self.theta.v0_m[i],torch.sqrt(self.var_v0)) for i in range(d)]
        xi = [rd.gauss(0, torch.exp(self.theta.logvar_xi/2)) for i in range(self.nb_patients)]
        tau = [rd.gauss(0, torch.exp(self.theta.logvar_tau/2)) for i in range(self.nb_patients)]
        s = []
        for i in range(self.nb_patients):
            s_i = []
            for l in range(self.Ns):
                s_i.append(torch.tensor(rd.gauss(0, 1)))
            s.append(s_i)
        beta = []
        for l in range(self.Ns):
            beta_l = []
            for k in range(d - 1):
                beta_l.append(rd.gauss(self.theta.beta_m[l][k], torch.sqrt(self.var_beta)))
            beta.append(beta_l)

        self.z_array.append(latent(t0, p0, v0, xi, tau, s, beta))
        self.err = self.compute_err(t0, p0, v0, xi, tau, s, beta)

    def compute_err(self, z_t0, z_p0, z_v0, z_xi, z_tau, z_s, z_beta):
        # Calcule sum_{i,j} (yij - eta^w_i(gamma_0, psi_i(tij))**2
        geo = geodesic(z_t0, z_p0, z_v0)
        b = house_holder(z_v0)
        res = 0
        for i in range(self.nb_patients):
            alpha_i = torch.exp(z_xi[i])
            w_i = torch.zeros(d)
            for k in range(d-1):
                sum_sbeta = 0
                for l in range(self.Ns):
                    sum_sbeta += z_s[i][l]*z_beta[l][k]
                for j in range(d):
                    w_i[j] = w_i[j] + b[k][j]
            exp_para_i = exp_para(geo, w_i)
            for j in range(len(self.data_t[i])):
                t_ij = (alpha_i)*(self.data_t[i][j] - z_t0 - z_tau[i]) + z_t0
                exp_para_i_t_ij = exp_para_i(t_ij)
                for k in range(d):
                    res  = res + (self.data_y[i][j][k] - exp_para_i_t_ij[k])**2
        return res

    def Leapfrog(self, z, r, eps):
        z_t0 = z.t0.clone().detach().requires_grad_(True)
        z_p0 = torch.tensor(z.p0, requires_grad = True)
        z_v0 = torch.tensor(z.v0, requires_grad = True)
        z_xi = torch.tensor(z.xi, requires_grad = True)
        z_tau = torch.tensor(z.tau, requires_grad = True)
        z_s = torch.tensor(z.s, requires_grad = True)
        z_beta = torch.tensor(z.beta, requires_grad = True)

        L_z = - (self.nb_mesures * self.theta.logvar_eps)/2  - self.compute_err(z_t0, z_p0, z_v0, z_xi, z_tau, z_s, z_beta)/(2*torch.exp(self.theta.logvar_eps)) + prior(z_t0, self.theta.t0_m, self.var_t0, z_p0, self.theta.p0_m, self.var_p0, z_v0, self.theta.v0_m, self.var_v0, z_xi, self.theta.logvar_xi, z_tau, self.theta.logvar_tau, z_s, z_beta, self.theta.beta_m, self.var_beta)
        L_z.backward()


        gradient_L_z = latent(z_t0.grad, z_p0.grad, z_v0.grad, z_xi.grad, z_tau.grad, z_s.grad, z_beta.grad)
        gradient_L_z.mul(eps/2)

        r.add_latent(gradient_L_z)

        #z.show()
        #print("gradient:")
        #gradient_L_z.show()
        
        r.mul(eps)
        z.add_latent(r)
        r.mul(1/eps)

        z_t0 = z.t0.clone().detach().requires_grad_(True)
        z_p0 = torch.tensor(z.p0, requires_grad = True)
        z_v0 = torch.tensor(z.v0, requires_grad = True)
        z_xi = torch.tensor(z.xi, requires_grad = True)
        z_tau = torch.tensor(z.tau, requires_grad = True)
        z_s = torch.tensor(z.s, requires_grad = True)
        z_beta = torch.tensor(z.beta, requires_grad = True)

        L_z = -(self.nb_mesures * self.theta.logvar_eps)/2 - self.compute_err(z_t0, z_p0, z_v0, z_xi, z_tau, z_s, z_beta)/(2*torch.exp(self.theta.logvar_eps)) + prior(z_t0, self.theta.t0_m, self.var_t0, z_p0, self.theta.p0_m, self.var_p0, z_v0, self.theta.v0_m, self.var_v0, z_xi, self.theta.logvar_xi, z_tau, self.theta.logvar_tau, z_s, z_beta, self.theta.beta_m, self.var_beta)
        L_z.backward()

        gradient_L_z = latent(z_t0.grad, z_p0.grad, z_v0.grad, z_xi.grad, z_tau.grad, z_s.grad, z_beta.grad)
        gradient_L_z.mul(eps/2)
        
        r.add_latent(gradient_L_z)

        #print("after Leapfrog :")
        #z.show()

    def BuildTree(self, z, r, log_u, v, j, eps):
        Delta_max = 1000
        
        if j == 0:
            z_bis = z.clone()
            r_bis = r.clone()
            #print("Before Leapfrog : ")
            self.Leapfrog(z_bis, r_bis, v*eps)

            L_z_bis = -(self.nb_mesures * self.theta.logvar_eps)/2 - self.compute_err(z_bis.t0, z_bis.p0, z_bis.v0, z_bis.xi, z_bis.tau, z_bis.s, z_bis.beta)/(2*torch.exp(self.theta.logvar_eps)) + prior(z_bis.t0, self.theta.t0_m, self.var_t0, z_bis.p0, self.theta.p0_m, self.var_p0, z_bis.v0, self.theta.v0_m, self.var_v0, z_bis.xi, self.theta.logvar_xi, z_bis.tau, self.theta.logvar_tau, z_bis.s, z_bis.beta, self.theta.beta_m, self.var_beta)
            r_bis_norm = r_bis.norm()
            n_bis = indic(log_u <= L_z_bis - r_bis_norm/2)
            s_bis = indic(L_z_bis - r_bis_norm/2 > log_u - Delta_max)

            return z_bis.clone(), r_bis.clone(), z_bis.clone(), r_bis.clone(), z_bis.clone(), n_bis, s_bis

        else:
            z_minus, r_minus, z_plus, r_plus, z_bis, n_bis, s_bis = self.BuildTree(z, r, log_u, v, j-1, eps)

            if s_bis == 1:
                if (v == -1):
                    #print("v = " + str(v))
                    z_minus_bis, r_minus_bis, _, _, z_terce_bis, n_terce, s_terce = self.BuildTree(z_minus, r_minus, log_u, v, j-1, eps)
                    z_minus = z_minus_bis.clone()
                    r_minus = r_minus_bis.clone()
                else:
                    #print("v = " + str(v))
                    _, _, z_plus_bis, r_plus_bis, z_terce_bis, n_terce, s_terce = self.BuildTree(z_plus, r_plus, log_u, v, j-1, eps)
                    z_plus = z_plus_bis.clone()
                    r_plus = r_plus_bis.clone()
                if (n_terce > 0) and (rd.random() <= n_terce/(n_bis + n_terce)):
                    z_bis.set(z_terce_bis)

                sum1 = torch.tensor(0.)
                #print("debug2")
                #z_plus.show()
                #z_minus.show()
                #r_plus.show()
                #r_minus.show()
                #print("end debug")
                sum1 += (z_plus.t0 - z_minus.t0)*r_minus.t0
                for i in range(d):
                    sum1 += (z_plus.p0[i] - z_minus.p0[i])*r_minus.p0[i]
                    sum1 += (z_plus.v0[i] - z_minus.v0[i])*r_minus.v0[i]
                for i in range(self.nb_patients):
                    sum1 += (z_plus.xi[i] - z_minus.xi[i])*r_minus.xi[i]
                    sum1 += (z_plus.tau[i] - z_minus.tau[i])*r_minus.tau[i]
                    for l in range(self.Ns):
                        sum1 += (z_plus.s[i][l] - z_minus.s[i][l])*r_minus.s[i][l]
                for l in range(self.Ns):
                    for k in range(d - 1):
                        sum1 += (z_plus.beta[l][k] - z_minus.beta[l][k])*r_minus.beta[l][k]


                sum2 = torch.tensor(0.)
                sum2 += (z_plus.t0 - z_minus.t0)*r_plus.t0
                for i in range(d):
                    sum2 += (z_plus.p0[i] - z_minus.p0[i])*r_plus.p0[i]
                    sum2 += (z_plus.v0[i] - z_minus.v0[i])*r_plus.v0[i]
                for i in range(self.nb_patients):
                    sum2 += (z_plus.xi[i] - z_minus.xi[i])*r_plus.xi[i]
                    sum2 += (z_plus.tau[i] - z_minus.tau[i])*r_plus.tau[i]
                    for l in range(self.Ns):
                        sum2 += (z_plus.s[i][l] - z_minus.s[i][l])*r_plus.s[i][l]
                for l in range(self.Ns):
                    for k in range(d - 1):
                        sum2 += (z_plus.beta[l][k] - z_minus.beta[l][k])*r_plus.beta[l][k]

                s_bis = s_terce * indic(sum1 >= 0)*indic(sum2 >= 0)
                n_bis = n_bis + n_terce
                
            return z_minus.clone(), r_minus.clone(), z_plus.clone(), r_plus.clone(), z_bis.clone(), n_bis, s_bis

    def update_z(self, eps):
        z_prev = self.z_array[-1]

        L = -(self.nb_mesures * self.theta.logvar_eps)/2 - self.err/(2*torch.exp(self.theta.logvar_eps)) + prior(z_prev.t0, self.theta.t0_m, self.var_t0, z_prev.p0, self.theta.p0_m, self.var_p0, z_prev.v0, self.theta.v0_m, self.var_v0, z_prev.xi, self.theta.logvar_xi, z_prev.tau, self.theta.logvar_tau, z_prev.s, z_prev.beta, self.theta.beta_m, self.var_beta)

        r0 = randomize(self.nb_patients, self.Ns, 1, 1, 1, 0, 0, 1, 1)
        log_u = torch.log(torch.tensor(rd.random())) + (L - r0.norm())
        z_minus = z_prev.clone()
        z_plus = z_prev.clone()
        z = z_prev.clone()
        r_minus = r0.clone()
        r_plus = r0.clone()
        j = 0
        n = 1
        s = 1

        while (s == 1 and self.limit_j >= j):
            print("j : " + str(j))
            vj = 2*int(2*rd.random()) - 1
            if (vj == -1):
                z_minus, r_minus, _, _, z_bis, n_bis, s_bis = self.BuildTree(z_minus, r_minus, log_u, vj, j, eps)
            else:
                _, _, z_plus, r_plus, z_bis, n_bis, s_bis = self.BuildTree(z_plus, r_plus, log_u, vj, j, eps) 
            if (s_bis == 1):
                if (rd.random() <= min(1, n_bis/n)):
                    z.set(z_bis)
            n = n + n_bis
            
            #z_plus.show()
            #z_minus.show()

            sum1 = torch.tensor(0.)
            sum1 += (z_plus.t0 - z_minus.t0)*r_minus.t0
            for i in range(d):
                sum1 += (z_plus.p0[i] - z_minus.p0[i])*r_minus.p0[i]
                sum1 += (z_plus.v0[i] - z_minus.v0[i])*r_minus.v0[i]
            for i in range(self.nb_patients):
                sum1 += (z_plus.xi[i] - z_minus.xi[i])*r_minus.xi[i]
                sum1 += (z_plus.tau[i] - z_minus.tau[i])*r_minus.tau[i]
                for l in range(self.Ns):
                    sum1 += (z_plus.s[i][l] - z_minus.s[i][l])*r_minus.s[i][l]
            for l in range(self.Ns):
                for k in range(d - 1):
                    sum1 += (z_plus.beta[l][k] - z_minus.beta[l][k])*r_minus.beta[l][k]


            sum2 = torch.tensor(0.)
            sum2 += (z_plus.t0 - z_minus.t0)*r_plus.t0
            for i in range(d):
                sum2 += (z_plus.p0[i] - z_minus.p0[i])*r_plus.p0[i]
                sum2 += (z_plus.v0[i] - z_minus.v0[i])*r_plus.v0[i]
            for i in range(self.nb_patients):
                sum2 += (z_plus.xi[i] - z_minus.xi[i])*r_plus.xi[i]
                sum2 += (z_plus.tau[i] - z_minus.tau[i])*r_plus.tau[i]
                for l in range(self.Ns):
                    sum2 += (z_plus.s[i][l] - z_minus.s[i][l])*r_plus.s[i][l]
            for l in range(self.Ns):
                for k in range(d - 1):
                    sum2 += (z_plus.beta[l][k] - z_minus.beta[l][k])*r_plus.beta[l][k]

            #print("The Booleans")
            #print(s_bis)
            #print(sum1)
            #print(sum2)
            s = s_bis*indic(sum1 >= 0)*indic(sum2 >= 0)
            j = j + 1

        self.z_array.append(z)
        self.err = self.compute_err(z.t0, z.p0, z.v0, z.xi, z.tau, z.s, z.beta)
        

    def maximise_theta(self):
        t0_mean = self.Sk[0].clone().detach()
        p0_mean = [self.Sk[1][i].clone().detach() for i in range(d)]
        v0_mean = [self.Sk[2][i].clone().detach() for i in range(d)]
        logvar_tau = torch.log(-2*self.Sk[3]/self.nb_patients).clone().detach()
        logvar_xi = torch.log(-2*self.Sk[4]/self.nb_patients).clone().detach()
        logvar_eps = torch.log(-2*self.Sk[5]/(self.nb_mesures)).clone().detach()
        beta_mean = []
        for l in range(self.Ns):
            beta_mean_l = []
            for k in range(d - 1):
                beta_mean_l.append(self.Sk[7][l][k].clone().detach())
            beta_mean.append(beta_mean_l)
        

        self.theta = parametres(t0_mean, p0_mean, v0_mean, beta_mean, logvar_xi, logvar_tau, logvar_eps)

    def compute_S(self, z):
        sum_tau2 = 0
        sum_xi2 = 0
        for i in range(self.nb_patients):
            sum_tau2 += z.tau[i]**2
            sum_xi2 += z.xi[i]**2
        sum_beta2 = 0
        for l in range(self.Ns):
            for k in range(d - 1):
                sum_beta2 += z.beta[l][k]**2
        
        return [z.t0, z.p0, z.v0, -sum_tau2/2, -sum_xi2/2, -self.err/2, -sum_beta2/2, z.beta]

    def update_Sk(self, Nb, beta):
        S_aux = self.compute_S(self.z_array[-1])
        eps = epsilon(Nb, beta, len(self.z_array))

        temp = []
        temp.append((1 - eps)*self.Sk[0] + eps*S_aux[0])

        new_p0 = []
        new_v0 = []
        for i in range(d):
            new_p0.append((1 - eps)*self.Sk[1][i] + eps*S_aux[1][i])
            new_v0.append((1 - eps)*self.Sk[2][i] + eps*S_aux[2][i])
        temp.append(new_p0)
        temp.append(new_v0)

        temp.append((1 - eps)*self.Sk[3] + eps*S_aux[3])
        temp.append((1 - eps)*self.Sk[4] + eps*S_aux[4])
        temp.append((1 - eps)*self.Sk[5] + eps*S_aux[5])
        temp.append((1 - eps)*self.Sk[6] + eps*S_aux[6])

        new_beta = []
        for l in range(self.Ns):
            new_beta_l = []
            for k in range(d - 1):
                new_beta_l.append((1 - eps)*self.Sk[7][l][k] + eps*S_aux[7][l][k])
            new_beta.append(new_beta_l)
        temp.append(new_beta)

        self.Sk = temp

        

def NUTS(data_t, data_y, theta_init, var_t0, var_p0, var_v0, var_beta, nb_patients, nb_mesures, eps, N, Nb, beta, limit_j):
    nuts = MCMC(data_t, data_y, theta_init, var_t0, var_p0, var_v0, var_beta, nb_patients, nb_mesures, limit_j)
    init_timer = default_timer()
    liste_theta = [theta_init.logvar_eps]
    liste_timer = [0.]

    timer_max = 3600*5
    i = 0

    nuts.init_z()
    while (i < N and default_timer() - init_timer < timer_max):
        print("itération : " +str(i + 1))
        nuts.theta.show()
        nuts.update_z(eps)
        nuts.update_Sk(Nb, beta)
        nuts.maximise_theta()
        liste_timer.append(default_timer() - init_timer)
        liste_theta.append(nuts.theta.logvar_eps)
        i += 1
    print("oooooooooooooooooooooooooooo")
    print("")
    print("parametres obtenus :")
    print("")
    nuts.theta.show()
    print("")
    '''
    for i in range(N):
        if ((i+1)%25 == 0):
            nuts.z_array[i].show()
    '''
    return nuts.theta, liste_theta, liste_timer

def NUTS_light(data_t, data_y, theta_init, step_size, N, Nb, limit_j):

    nb_patients = len(data_t)
    nb_mesures = 0
    for i in range(nb_patients):
        nb_mesures += len(data_t[i])
    
    power = 0.65

    var_t0 = torch.tensor(0.05**2)
    var_p0 = torch.tensor(0.1**2)
    var_v0 = torch.tensor(0.025**2)
    var_beta = torch.tensor(0.025**2)

    return NUTS(data_t, data_y, theta_init, var_t0, var_p0, var_v0, var_beta, nb_patients, nb_mesures, step_size, N, Nb, power, limit_j)

def generate_data(data_t, th, var_t0, var_p0, var_v0, var_beta):
    p0 = [rd.gauss(th.p0_m[i], torch.sqrt(var_p0)) for i in range(d)]
    v0 = [rd.gauss(th.v0_m[i], torch.sqrt(var_v0)) for i in range(d)]
    t0 = rd.gauss(th.t0_m, torch.sqrt(var_t0))
    Ns = th.Ns
    geo = geodesic(t0, p0, v0)
    print(t0, p0, v0)
    b = house_holder(torch.tensor(v0))
    beta = []
    for l in range(Ns): 
        beta_l = []
        for k in range(d - 1):
            beta_l.append(rd.gauss(th.beta_m[l][k], torch.sqrt(var_beta)))
        beta.append(beta_l)
    y = []
    for i in range(len(data_t)):
        xi_i = rd.gauss(0, torch.exp(th.logvar_xi/2))
        tau_i = rd.gauss(0, torch.exp(th.logvar_tau/2))
        alpha_i = torch.exp(xi_i)
        s_i = []
        for l in range(Ns):
            s_i.append(rd.gauss(0, 1))
        w_i = torch.zeros(d)
        for k in range(d-1):
            sum_sbeta = 0
            for l in range(Ns):
                sum_sbeta += s_i[l]*beta[l][k]
            for j in range(d):
                w_i[j] = w_i[j] + sum_sbeta*b[k][j]
        y_i = []
        for j in range(len(data_t[i])):
            eps_ij = [rd.gauss(0, torch.exp(th.logvar_eps/2)) for i in range(d)]
            t_ij = alpha_i*(data_t[i][j] - t0 - tau_i) + t0
            exp_para_geo_w_i_t_ij = exp_para(geo, w_i)(t_ij)
            y_ij = [exp_para_geo_w_i_t_ij[k] + eps_ij[k]  for k in range(d)]
            y_i.append(y_ij)
        y.append(y_i)
    return y, geo

def plot_data(data_t, data_y, coord):
    for i in range(len(data_t)):
        t = np.array([data_t[i][j] for j in range(len(data_t[i]))])
        y = np.array([data_y[i][j][coord] for j in range(len(data_y[i]))])
        plt.plot(t, y, '-')

def plot_geo(times, geo, coord):
    data_time = np.array([times[i] for i in range(len(times))])
    y = np.array([geo(times[i])[coord] for i in range(len(times))])
    plt.plot(data_time, y, '-', linewidth = 3, color = 'black')

########## hyper-paramètres 

var_t0 = torch.tensor(0.05**2)
var_p0 = torch.tensor(0.1**2)
var_v0 = torch.tensor(0.025**2)
var_beta = torch.tensor(0.025**2)

########## Nb patients et mesures

#nb_patients = 12
#nb_mesures = 8 * nb_patients

########## paramètres initiaux

sigma_xi = torch.tensor(0.5)
sigma_tau = torch.tensor(0.1)
sigma_eps = torch.tensor(0.005)

sigma_xi_init = torch.tensor(0.67)
sigma_tau_init = torch.tensor(0.042)
sigma_eps_init = torch.tensor(1.5)

t0_m = torch.tensor(0.5)
p0_m = [torch.tensor(0.5), torch.tensor(0.5)]
v0_m = [torch.tensor(1.), torch.tensor(2.)]

t0_m_init = torch.tensor(0.43)
p0_m_init = [torch.tensor(0.4), torch.tensor(0.6)]
v0_m_init = [torch.tensor(0.8), torch.tensor(1.5)]

Ns = 1
beta_m = [[torch.tensor(0.)]]
beta_m_init = [[torch.tensor(0.5)]]


theta = parametres(t0_m, p0_m, v0_m, beta_m, 2*torch.log(sigma_xi), 2*torch.log(sigma_tau), 2*torch.log(sigma_eps))
theta_init = parametres(t0_m_init, p0_m_init, v0_m_init, beta_m_init, 2*torch.log(sigma_xi_init), 2*torch.log(sigma_tau_init), 2*torch.log(sigma_eps_init))

########## Pour affichage

#step_t = 1/((nb_mesures//nb_patients) - 1)
#data_t = [[i*step_t for i in range(nb_mesures//nb_patients)] for j in range(nb_patients)]
#
#data_y, geo = generate_data(data_t, theta, var_t0, var_p0, var_v0, var_beta)

nb_patients = 40
prev_data_t, data_y = read("test.csv", 2, nb_patients)

data_t = normalize_temp(prev_data_t)

nb_mesures = 0
for i in range(len(data_t)):
    nb_mesures += len(data_t[i])

########## Paramètres algorithme

eps = 0.003 # step_size
N = 1000 # nb d'itérations
Nb = 40 # à partir de quand on prend en compte les z dans Sk
power = 0.65 # la puissance de la suite epsilon_k

limit_j = 9 # le nombre maximum de tour de boucles par itération de NUTS, (pour pas que ça prenne trop longtemps)


theta_nuts, liste_theta, liste_timer = NUTS(data_t, data_y, theta_init, var_t0, var_p0, var_v0, var_beta, nb_patients, nb_mesures, eps, N, Nb, power, limit_j)

########## Pour affichage

data_y_nuts, geo_nuts = generate_data(data_t, theta_nuts, var_t0, var_p0, var_v0, var_beta)

for i in range(d):
    plt.figure(i + 1)
    plot_data(data_t, data_y, i)
    #plot_geo(data_t[0], geo, i)

for i in range(d):
    plt.figure(i + d + 1)
    plot_data(data_t, data_y_nuts, i)
    plot_geo(data_t[0], geo_nuts, i)

plt.figure(10)

plt.title("logarithme du terme d'erreur (bruit du modèle)")
plt.xlabel("Temps pris (secondes)")

plt.plot(np.array(liste_timer), np.array(liste_theta), '-', color = "blue")



plt.show()














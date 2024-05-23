# M = ]0, 1[
import numpy as np
import matplotlib.pyplot as plt
import random as rd

def g(p):
    return 1/(p*p*(1-p)*(1-p))

def geodesic(t, p, v):
    C = v/(p*(1-p))
    A = np.exp(C*t)*(1 - p)/p
    return C, A

def compute_geodesic(C, A, t):
    return 1/(1 + A*np.exp(-C*t))

def parallel_transport(t, v, C, A):
    p = compute_geodesic(C, A, t)
    return v/(p*(1-p))

def compute_parallel_transport(C, p):
    return C*(p*(1-p))

def exponential(p, v):
    C, A = geodesic(0, p, v)
    return compute_geodesic(C, A, 1)

def exp_parallelization(C, A, v, t_0, t):
    C_pt = parallel_transport(t_0, v, C, A)
    gamma_t = compute_geodesic(C, A, t)
    return exponential(gamma_t, compute_parallel_transport(C_pt, gamma_t))

def plot_data(data_t, data_y):
    for i in range(len(data_t)):
        t = np.array([data_t[i][j] for j in range(len(data_t[i]))])
        y = np.array([data_y[i][j] for j in range(len(data_y[i]))])
        plt.plot(t, y, '-')

def generate_data(data_t, p0m, p0v, v0m, v0v, t0m, t0v, wm, wv, ksiv, tauv, var):
    p0 = rd.gauss(p0m, p0v)
    v0 = rd.gauss(v0m, v0v)
    t0 = rd.gauss(t0m, t0v)
    C0, A0 = geodesic(t0, p0, v0)
    y = []
    for i in range(len(data_t)):
        w_i = rd.gauss(wm, wv)
        ksi_i = rd.gauss(0, ksiv)
        tau_i = rd.gauss(0, tauv)
        alpha_i = np.exp(ksi_i)
        y_i = []
        for j in range(len(data_t[i])):
            eps_ij = rd.gauss(0, var)
            t_ij = alpha_i*(data_t[i][j] - t0 - tau_i) + t0
            y_ij = exp_parallelization(C0, A0, w_i, t0, t_ij) + eps_ij
            y_i.append(y_ij)
        y.append(y_i)
    return y

def epsilon(Nb, beta, k):
    if k > Nb:
        return pow(k-Nb, -1 * beta)
    else:
        return 1

def q_cond_z(data_t, data_y, t0, C0, A0, w, ksi, tau):
    # Calcule sum_{i,j} (yij - eta^w_i(gamma_0, psi_i(tij))**2
    res = 0
    for i in range(len(data_t)):
        for j in range(len(data_t[i])):
            t_ij = (np.exp(ksi[i]))*(data_t[i][j] - t0 - tau[i]) + t0
            '''print(C0, A0, w[i], t0, t_ij)'''
            res  = res + (data_y[i][j] - exp_parallelization(C0, A0, w[i], t0, t_ij))**2
    return res

def q_cond_z_i(data_t_i, data_y_i, t0, C0, A0, w_i, ksi_i, tau_i):
    # Calcule sum_{i,j} (yij - eta^w_i(gamma_0, psi_i(tij))**2
    res = 0
    for j in range(len(data_t_i)):
        t_ij = (np.exp(ksi_i))*(data_t_i[j] - t0 - tau_i) + t0
        res  = res + (data_y_i[j] - exp_parallelization(C0, A0, w_i, t0, t_ij))**2
    return res

def S(data_t, data_y, p0, t0, v0, C0, A0, w, ksi, tau, sum_ij):
    res = []
    res.append(-p0**2/2)
    res.append(-t0**2/2)
    res.append(-v0**2/2)
    res.append(p0)
    res.append(t0)
    res.append(v0)

    sumw = 0
    sumtau2 = 0
    sumksi2 = 0
    sumw2 = 0

    for i in range(len(data_t)):
        sumw += w[i]
        sumtau2 += tau[i]**2
        sumksi2 += ksi[i]**2
        sumw2 += w[i]**2

    res.append(-sumtau2/2)
    res.append(-sumksi2/2)
    res.append(-sumw2/2)
    res.append(sumw)
    res.append(-sum_ij/2)

    return res
    

def MCMC_SAEM(data_t, data_y, Nb, beta, N, p0m_init, v0m_init, t0m_init, wm_init, ksiv_init, tauv_init, var_init):

    n = len(data_t)
    p0v = 0.1
    v0v = 0.025
    t0v = 0.5
    wv = 0.005

    nb_pop = 0
    nb_indiv = [0 for i in range(n)]

    Nij = 0
    for i in range(n):
        Nij += len(data_t[i])
    
    # Paramètres à faire varier
    
    zeta_p0 = 0.1
    zeta_v0 = 0.025
    zeta_t0 = 0.5
    zeta_ksi = 0.1
    zeta_tau = 0.5
    zeta_w = 0.005

    # Initialisation

    p0m = p0m_init
    v0m = v0m_init
    t0m = t0m_init
    wm = wm_init
    ksiv = ksiv_init
    tauv = tauv_init
    var = var_init
    p0 = rd.gauss(p0m, p0v)
    v0 = rd.gauss(v0m, v0v)
    t0 = rd.gauss(t0m, t0v)
    C0, A0 = geodesic(t0, p0, v0)

    w = [0 for i in range(n)]
    ksi = [0 for i in range(n)]
    tau = [0 for i in range(n)]
    
    for i in range(n):
        w[i] = rd.gauss(wm, zeta_w)
        ksi[i] = rd.gauss(0, zeta_ksi)
        tau[i] = rd.gauss(0, zeta_tau)

    print("before")
    sum_tab = [q_cond_z_i(data_t[i], data_y[i], t0, C0, A0, w[i], ksi[i], tau[i]) for i in range(n)]
    print("after")
    sum_tab_star = [sum_tab[i] for i in range(n)]
    sum_ij = 0
    sum_ij_star = 0
    for i in range(n):
        sum_ij += sum_tab[i]
        sum_ij_star += sum_tab_star[i]
    Sk = S(data_t, data_y, p0, t0, v0, C0, A0, w, ksi, tau, sum_ij)

    for k in range(N):

        # Calcul de z^(k) avec Metropolis-Hastings

        print(k)

        p0star = rd.gauss(p0, zeta_p0)
        v0star = rd.gauss(v0, zeta_v0)
        t0star = rd.gauss(t0, zeta_t0)
        C0star, A0star = geodesic(t0star, p0star, v0star)
        sum_ij_star = 0
        for i in range(n):
            sum_tab_star[i] = q_cond_z_i(data_t[i], data_y[i], t0star, C0star, A0star, w[i], ksi[i], tau[i])
            sum_ij_star += sum_tab_star[i]

        y = sum_ij_star/(-2 * var**2) - (p0star - p0m)**2/(2*p0v**2) - (v0star - v0m)**2/(2*v0v**2) - (t0star - t0m)**2/(2*t0v**2)
        x = sum_ij/(2 * var**2) + (p0 - p0m)**2/(2*p0v**2) + (v0 - v0m)**2/(2*v0v**2) + (t0 - t0m)**2/(2*t0v**2)

        if (y + x) > 0:
            alpha_pop = 1
        else:
            alpha_pop = np.exp(y + x)
        u = rd.random()
        if (u <= alpha_pop):
            '''
            print("accept_pop")
            print(var, p0v, v0v, t0v)
            print(sum_ij, sum_ij_star)
            print(x, y)
            print("alpha_pop : " + str(alpha_pop))
            print("p0, v0, t0, C0, A0 :")
            print(p0, v0, t0, C0, A0)
            '''
            p0 = p0star
            v0 = v0star
            t0 = t0star
            C0 = C0star
            A0 = A0star
            for l in range(n):
                sum_tab[l] = sum_tab_star[l]
                sum_ij = sum_ij_star
            #print(p0, v0, t0, C0, A0)
            nb_pop += 1
            

        for i in range(n):
            wstar = rd.gauss(w[i], zeta_w)
            ksistar = rd.gauss(ksi[i], zeta_ksi)
            taustar = rd.gauss(tau[i], zeta_tau)

            sum_ij_star = sum_ij_star - sum_tab_star[i]
            sum_tab_star[i] = q_cond_z_i(data_t[i], data_y[i], t0, C0, A0, wstar, ksistar, taustar)
            sum_ij_star = sum_ij_star + sum_tab_star[i]

            x = sum_ij/(-2 * var**2) - (w[i])**2/(2*wv**2) - (ksi[i])**2/(2*ksiv**2) - (tau[i])**2/(2*tauv**2)
            y = sum_ij_star/(-2 * var**2) - (wstar)**2/(2*wv**2) - (ksistar)**2/(2*ksiv**2) - (taustar)**2/(2*tauv**2)

            if (y > x):
                alpha_i = 1
            else:
                alpha_i = np.exp(y - x)
                
            u = rd.random()
            if (u <= alpha_i):
                w[i] = wstar
                ksi[i] = ksistar
                tau[i] = taustar
                sum_tab[i] = sum_tab_star[i]
                sum_ij = sum_ij_star
                #print("accept_" + str(i))
                #print("alpha_" + str(i) + " : " + str(alpha_i))
                nb_indiv[i] += 1

        # Calcul de S_k

        Saux = S(data_t, data_y, p0, t0, v0, C0, A0, w, ksi, tau, sum_ij)
        eps = epsilon(Nb, beta, k)

        for i in range(len(Sk)):
            Sk[i] = Sk[i] + eps*(Saux[i] - Sk[i])

        # Calcul de theta^(k)
        
        p0m = Sk[3]
        v0m = Sk[5]
        t0m = Sk[4]
        wm = Sk[9]/n
        tauv = np.sqrt(-2* Sk[6]/n)
        ksiv = np.sqrt(-2 * Sk[7]/n)
        var = np.sqrt(-2 * Sk[10]/Nij)

        #print([p0m, v0m, t0m, wm, ksiv, tauv, var])

    print("nb_pop : " + str(nb_pop))
    for i in range(n):
        print("nb_indiv[" + str(i) + "] : " + str(nb_indiv[i]))

    return [p0m, v0m, t0m, wm, ksiv, tauv, var]


        
T = 10
step_g = 0.01
N_g = int(T//step_g + 1)

n = 10

t_0 = 5
v = 0.25

C_g, A_g = geodesic(t_0, 0.5, 0.4)

t_g = np.array([i*step_g for i in range(N_g)])
y_g = np.array([compute_geodesic(C_g, A_g, t_g[i]) for i in range(N_g)])

plt.figure()

plt.plot(t_g, y_g, '-', linewidth = 3, color = 'black')

step_t = 0.2
N_t = int(T//step_t + 1)
data_t = [[j*step_t for j in range(N_t)] for i in range(n)]
p0m, p0v = 0.5, 0.1
v0m, v0v = 0.4, 0.025
t0m, t0v = 5, 0.5
wm, wv = 0, 0.005
ksiv = 0.4
tauv = 0.75
var = 0
data_y = generate_data(data_t, p0m, p0v, v0m, v0v, t0m, t0v, wm, wv, ksiv, tauv, var)

plot_data(data_t, data_y)

Nb = 100
N = 2000
beta = 0.65
p0m_init, v0m_init, t0m_init, wm_init, ksiv_init, tauv_init, var_init = 0.5, 0.2, 4, 0.5, 1, 0.5, 0.0015

theta_MCMCSAEM = MCMC_SAEM(data_t, data_y, Nb, beta, N, p0m_init, v0m_init, t0m_init, wm_init, ksiv_init, tauv_init, var_init)
print(theta_MCMCSAEM)
print([p0m, v0m, t0m, wm, ksiv, tauv, var])

plt.show()

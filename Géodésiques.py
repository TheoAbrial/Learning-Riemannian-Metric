import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


#Première version : résolution d'EDO

coef = np.array([-1])

#Polynôme définissant la métrique g = 1/P^2
P = lambda x : sum([coef[i]*x**(i+2) for i in range(len(coef))]) - sum(coef)*x

def geodesic(t_0, T, N, p_0, v_0):
    #Résoud l'équation des géodésiques sur l'intervalle [t_0-T, t_0+T] avec y(t_0)=p_0

    F = lambda y,t : v_0*P(y)/P(p_0)
        
    t = np.linspace(t_0, t_0+T, N)
        
    y = sc.integrate.odeint(F, p_0, t)
    y_rev = sc.integrate.odeint(F, p_0, -t+2*t_0)
        
    return np.concatenate((np.flip(-t+2*t_0),t)), np.concatenate((np.flip(y_rev), y))


def parallel_transport(t_0, T, N, p_0, v_0, w):
    #Calcule le transport parallèle de w le long d'une géodésique
    t, g = geodesic(t_0, T, N, p_0, v_0)
    return w*P(g)/P(p_0)

def exp_parallelization(t_0, T, N, p_0, v_0, w):
    time, g = geodesic(t_0, T, N, p_0, v_0) #La géodésique à exp-paralléliser
    w_t = w*P(g)/P(p_0) #Le transport parallèle de w le long de la géodésique
    exp_p = np.zeros_like(g)
    for i in range(g.shape[0]):
        t, g_t = geodesic(0, 1, 100, g[i], w_t[i])
        exp_p[i] = g_t[-1]
    return exp_p

#Deuxiéme version : obtenir les géodésiques à partir d'une seule

def geodesic_2(N, h, p_0, v_0, t_1, p_1, v_1):
    #On construit une géodésique 
    F = lambda y,t : v_0*P(y)/P(p_0)
    t = np.linspace(0, N*h, N)
    g_0 = sc.integrate.odeint(F, p_0, t)
    
    #On reparamétrise la géodésique pour en obtenir une autre
    data_g = np.array([])
    data_t = np.array([])
    
    b = t[np.argmin(np.abs(g_0-p_1))]
    a = v_1*P(p_0)/(v_0*P(p_1))
    
    for time in t:
        t_new = a*time+b
        if t_new < t[-1]:
            ind = np.argmin(np.abs(t-t_new))
            data_t = np.append(data_t, time+t_1)
            data_g = np.append(data_g, g_0[ind])
    
    return data_t, data_g

def exp_parallelization_2(N, h, p_0, v_0, t_1, p_1, v_1, w):
    t,g = geodesic_2(N, h, p_0, v_0, t_1, p_1, v_1)
    return t+w/v_1, g
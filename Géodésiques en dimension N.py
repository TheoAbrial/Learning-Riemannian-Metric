import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

"""
Résolution de l'équation des géodésiques en dimension N.
"""

N = 5 #dimension de la variété

#Première version : intégration des équations de Hamilton.
#On considérera la métrique diagonale.
 
   
def F(t, y, g, grad_g):
    x = y[:N]
    alpha = y[N:]
    
    metric = g(x) #metric[i]=g_ii(x)
    grad_metric = grad_g(x) #grad_metric[i,j] = dg_jj/dx_i
    
    res_1 = np.array([alpha[i]/metric[i] for i in range(N)])
    res_2 = np.array([sum(alpha[j]**2*grad_metric[i,j]/metric[j]**2 for j in range(N))/2 for i in range(N)])
    
    return np.concatenate((res_1, res_2))
  
    
def geo_hamilton(g, grad_g, t_0, t_f, p_0, v_0):
    #Résoud les équations de Hamilton.
    #Renvoie la gédésique y telle que y(t_0)=p_0 et y'(t_0) = v_0.
    
    metric_0 = g(p_0)
    alpha_0 = metric_0*v_0
    
    y_0 = np.concatenate(p_0, alpha_0)
    
    return sc.integrate.solve_ivp(F, [t_0, t_f], y_0, args=(g, grad_g))
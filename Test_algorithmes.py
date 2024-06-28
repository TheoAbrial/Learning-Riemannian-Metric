from Visualisation_dim1 import *
from Nuts_Logistic_dim1 import *
import matplotlib.pyplot as plt
import numpy as np


########## hyper-paramètres 

var_t0 = torch.tensor(0.05**2)
var_p0 = torch.tensor(0.1**2)
var_v0 = torch.tensor(0.025**2)

########## Nb patients et mesures

nb_patients = 15
nb_mesures = 8*15

########## paramètres initiaux

sigma_xi = torch.tensor(0.5)
sigma_tau = torch.tensor(0.1)
sigma_eps = torch.tensor(0.005)

sigma_xi_init = torch.tensor(0.67)
sigma_tau_init = torch.tensor(0.042)
sigma_eps_init = torch.tensor(0.1)

t0_m = torch.tensor(0.5)
p0_m = torch.tensor(0.5)
v0_m = torch.tensor(3)

t0_m_init = torch.tensor(0.43)
p0_m_init = torch.tensor(0.4)
v0_m_init = torch.tensor(2.4)

theta = parametres(t0_m, p0_m, v0_m, 2*torch.log(sigma_xi), 2*torch.log(sigma_tau), 2*torch.log(sigma_eps))
theta_init = parametres(t0_m_init, p0_m_init, v0_m_init, 2*torch.log(sigma_xi_init), 2*torch.log(sigma_tau_init), 2*torch.log(sigma_eps_init))


########## Pour affichage

'''

# Données Synthétiques 

step_t = 1/((nb_mesures//nb_patients) - 1)
data_t = [[i*step_t for i in range(nb_mesures//nb_patients)] for j in range(nb_patients)]

data_y, geo = generate_data(data_t, theta, var_t0, var_p0, var_v0)
#'''

# Données Réelles 

prev_data_t, data_y = read("test.csv", 1, 33)

data_t = normalize_temp(prev_data_t)


nb_train = 30
data_t_train = data_t[:nb_train]
data_y_train = data_y[:nb_train]
data_t_incomplete = [[liste[i] for i in range(2)] for liste in data_t[nb_train:]]
data_y_incomplete = [[liste[i] for i in range(2)] for liste in data_y[nb_train:]]
data_t_complete = data_t[nb_train:]
data_y_complete = data_y[nb_train:]


h = 0.01
nbre = 100
data_t_courbe = [h*i for i in range(nbre + 1)]


step_size = 0.0025 # step_size
N = 20# nb d'itérations
Nb = 10 # à partir de quand on prend en compte les z dans Sk
limit_j = 7 # le nombre maximum de tour de boucles par itération de NUTS, (pour pas que ça prenne trop longtemps)

param_nuts = NUTS_parameters(N, Nb, step_size, limit_j)

nuts, data_y_computed, data_y_courbe = complete_disease_NUTS(data_t_train, data_t_incomplete, data_t_complete, data_t_courbe, data_y_train, data_y_incomplete, param_nuts, theta_init)

compare_data(data_t_complete, data_y_complete, data_t_courbe ,data_y_courbe)

plt.show()
















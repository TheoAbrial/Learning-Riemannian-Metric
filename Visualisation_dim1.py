from Nuts_Euclidian_dim1 import *
from CSV_Reader import *
import matplotlib.pyplot as plt
import numpy as np

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

class NUTS_parameters:
    def __init__(self, N, Nb, step_size, limit_j):
        self.N = N
        self.Nb = Nb
        self.step_size = step_size
        self.limit_j = limit_j

def complete_disease_NUTS(data_t_train, data_t_incomplete, data_t_complete, data_y_train, data_y_incomplete, param_nuts, theta_init):
    # les train sont seulement pour s'entraîner
    # les incomplete sont les débuts des mesures des patients dont on doit prédire la progression de la maladie
    # data_t_complete comprend data_t_incomplete et le reste des temps auxquels on doit prédire les valeurs
    
    data_t = []
    data_y = []

    nb_patients_train = len(data_t_train)
    nb_patients_incomplete = len(data_t_incomplete)

    for i in range(len(data_t_train)):
        data_t_i = []
        data_y_i = []
        for j in range(len(data_t_train[i])):
            data_t_i.append(data_t_train[i][j])
            data_y_i.append(data_y_train[i][j])
        data_t.append(data_t_i)
        data_y.append(data_y_i)
    
    for i in range(len(data_t_incomplete)):
        data_t_i = []
        data_y_i = []
        for j in range(len(data_t_incomplete[i])):
            data_t_i.append(data_t_incomplete[i][j])
            data_y_i.append(data_y_incomplete[i][j])
        data_t.append(data_t_i)
        data_y.append(data_y_i)

    nuts = NUTS_light(data_t, data_y, theta_init, param_nuts.step_size, param_nuts.N, param_nuts.Nb, param_nuts.limit_j)

    
    nb_moy = 10
    z = nuts.z_array[(-nb_moy):]
    entier = len(z[0].xi[nb_patients_train:])
    t0 = 0
    p0 = 0
    v0 = 0
    xi = [0 for i in range(entier) ]
    tau = [0 for i in range(entier) ]
    for i in range(nb_moy):
        t0 += z[i].t0/nb_moy
        p0 += z[i].p0/nb_moy
        v0 += z[i].v0/nb_moy
        for j in range(entier):
            xi[j] += z[i].xi[j]/nb_moy
            tau[j] += z[i].tau[j]/nb_moy
    '''
    z = nuts.z_array[-1]
    t0 = z.t0
    p0 = z.p0
    v0 = z.v0
    xi = z.xi[nb_patients_train:]
    tau = z.tau[nb_patients_train:]
    '''

    latent(t0, p0, v0, xi, tau).show()
    print(xi)

    geo = geodesic(t0, p0, v0)

    data_y_complete = []
    for i in range(nb_patients_incomplete):
        data_y_complete_i = []
        alpha_i = torch.exp(xi[i])
        for j in range(len(data_t_complete[i])):
            t_ij = alpha_i*(data_t_complete[i][j] - t0 - tau[i]) + t0
            y_ij = geo(t_ij)
            data_y_complete_i.append(y_ij)
        data_y_complete.append(data_y_complete_i)
    
    return nuts, data_y_complete

def compare_data(data_t, data_y, data_y_computed):
    nb_patients = len(data_t)

    for i in range(nb_patients):
        plt.figure(i + 1)
        nb_mesures = len(data_t[i])

        t = np.array([data_t[i][j] for j in range(nb_mesures)])
        y = np.array([data_y[i][j] for j in range(nb_mesures)])
        y_computed = np.array([data_y_computed[i][j] for j in range(nb_mesures)])

        plt.plot(t, y, '+', color = "black")
        plt.plot(t, y_computed, '-', color = "black")

        














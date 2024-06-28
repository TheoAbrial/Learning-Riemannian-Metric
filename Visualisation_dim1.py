from Nuts_Logistic_dim1 import *
from CSV_Reader import *
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer

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

def complete_disease_NUTS(data_t_train, data_t_incomplete, data_t_complete, data_t_courbe, data_y_train, data_y_incomplete, param_nuts, theta_init):
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

    
    nb_moy = param_nuts.N - param_nuts.Nb
    z= nuts.z_array[(-nb_moy):] 
    entier = len(z[0].xi[nb_patients_train:])
    t0 = torch.tensor(0.)
    p0 = torch.tensor(0.)
    v0 = torch.tensor(0.)
    xi = [torch.tensor(0.) for i in range(entier)]
    tau = [torch.tensor(0.) for i in range(entier)]
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

    def latent_to_liste(z):
        liste = []
        liste.append(float(z.t0))
        liste.append(float(z.p0))
        liste.append(float(z.v0))
        for i in range(len(z.xi)):
            liste.append(float(z.xi[i]))
        for i in range(len(z.tau)):
            liste.append(float(z.tau[i]))
        return liste
    
    def liste_to_latent(liste):
        n = (len(liste) - 3)//2
        z_t0 = torch.tensor(liste[0], requires_grad = True)
        z_p0 = torch.tensor(liste[1], requires_grad = True)
        z_v0 = torch.tensor(liste[2], requires_grad = True)
        z_xi = [torch.tensor(liste[3 + i], requires_grad = True) for i in range(n)]
        z_tau = [torch.tensor(liste[3 + n + i], requires_grad = True) for i in range(n)]
        return latent(z_t0, z_p0, z_v0, z_xi, z_tau)

    # Définir la fonction de Rosenbrock
    def rosenbrock(x):
        def compute_err(z_t0, z_p0, z_v0, z_xi, z_tau):
            
            data_y = nuts.data_y
            data_t = nuts.data_t
            geo = geodesic(z_t0, z_p0, z_v0)
            res = torch.tensor(0.)
            for i in range(entier):
                alpha_i = torch.exp(z_xi[i])
                for j in range(len(data_t[i])):
                    t_ij = (alpha_i)*(data_t[i][j] - z_t0 - z_tau[i]) + z_t0
                    res  = res + alpha_i + z_tau[i]#(data_y[i][j] - geo(t_ij))**2
            return res
        
        z = liste_to_latent(x)
        return compute_err(z.t0, z.p0, z.v0, z.xi, z.tau) #x[0], x[1], x[2], x[3:3 + n], x[-n:])

    # Gradient de la fonction de Rosenbrock
    def rosenbrock_grad(x):
        n = (len(x) - 3)//2 
        vect = torch.tensor(x, requires_grad = True)

        res = torch.tensor(0.)

        data_y = nuts.data_y
        data_t = nuts.data_t
        geo = geodesic(vect[0], vect[1], vect[2])
        res = torch.tensor(0.)
        for i in range(entier):
            alpha_i = torch.exp(vect[i + 3])
            for j in range(len(data_t[i])):
                t_ij = (alpha_i)*(data_t[i][j] - vect[0] - vect[i + 3 + n]) + vect[0]
                res  = res + (data_y[i][j] - geo(t_ij))**2

        res.backward()

        test = vect.grad

        grad = [float(test[i]) for i in range(len(vect))]
        grad[0] = 0.
        grad[1] = 0.
        grad[2] = 0.

        return np.array(grad)

    # Descente de gradient avec recherche linéaire pour un pas optimal
    def gradient_descent_optimal_step(x_init, tol=1e-7, max_sec=900, max_iter = 100000):
        x = x_init
        iter_count = 0
        path = [x_init]

        time_init = default_timer()

        def line_search(xk, pk, alpha_init=100000.0, c1=0.0001, c2=0.9, max_time=200):
            time_init = default_timer()
            alpha = alpha_init
            while True:
                new_x = xk + alpha * pk
                if rosenbrock(new_x) <= rosenbrock(xk) + c1 * alpha * np.dot(rosenbrock_grad(xk), pk)  or (default_timer() - time_init > max_time):
                    if np.dot(rosenbrock_grad(new_x), pk) >= c2 * np.dot(rosenbrock_grad(xk), pk) or (default_timer() - time_init > max_time):
                        break
                alpha *= 0.95  # Diminuer alpha
            return alpha

        while (default_timer() < max_sec + time_init and max_iter > iter_count):
            print(iter_count)
            grad = rosenbrock_grad(x)
            if np.linalg.norm(grad) < tol:
                break

            # Recherche linéaire pour trouver le pas optimal
            pk = -grad
            alpha = line_search(x, pk)

            x = x + alpha * pk
            path.append(x)
            iter_count += 1

        return x, iter_count, path

    zbis = latent(t0, p0, v0, xi, tau)

    #liste, iter_count, path = gradient_descent_optimal_step(latent_to_liste(zbis))

    #z = latent(liste[0], liste[1], liste[2], liste[3:3 + (len(liste) - 3)//2], liste[-(len(liste) - 3)//2:])

    z = zbis

    z.show()
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

    data_y_courbe = []
    for i in range(nb_patients_incomplete):
        data_y_courbe_i = []
        alpha_i = torch.exp(xi[i])
        for j in range(len(data_t_courbe)):
            t_ij = alpha_i*(data_t_courbe[j] - t0 - tau[i]) + t0
            y_ij = geo(t_ij)
            data_y_courbe_i.append(y_ij)
        data_y_courbe.append(data_y_courbe_i)
    
    return nuts, data_y_complete, data_y_courbe

def compare_data(data_t, data_y, data_t_courbe, data_y_courbe):
    nb_patients = len(data_t)

    for i in range(nb_patients):
        plt.figure(i + 1)
        nb_mesures = len(data_t[i])
        nb_mesures_courbes = len(data_t_courbe)

        t = np.array([data_t[i][j] for j in range(nb_mesures)])
        y = np.array([data_y[i][j] for j in range(nb_mesures)])
        t_courbe = np.array([data_t_courbe[j] for j in range(nb_mesures_courbes)])
        y_courbe = np.array([data_y_courbe[i][j] for j in range(nb_mesures_courbes)])

        plt.plot(t, y, '+', color = "black")
        plt.plot(t_courbe, y_courbe, '-', color = "black")

        














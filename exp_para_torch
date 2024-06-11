import torch
import torchode as to
import matplotlib.pyplot as plt


#g(x) = [g_11(x) , g_22(x),... , g_nn(x)]

def metric(param_metric):
    def g(x):
        return 1/x*(1-x)
    return g

dim_geo = 5 

def compute_F(param_metric):
    g = metric(param_metric)
    def jaco_g(x):
        input = torch.tensor(x,requires_grad=True)
        return torch.autograd.functional.jacobian(g,input) 

    def F(t,y):
        x = y[:dim_geo]
        alpha = y[dim_geo:]

        metric = g(x)
        jacobian = jaco_g(x)
        b =  -alpha**2/(2*metric**2)

        F_x = alpha/metric
        F_alpha = torch.linalg.matmul(jacobian,b)

        return torch.cat((F_x,F_alpha))

    return F

def geo_hamilton(param_metric, t, p0, v0):
    #Résoud les équations de Hamilton.
    #Renvoie la gédésique y telle que y(t_0)=p_0 et y'(t_0) = v_0.
    g = metric(param_metric)

    metric_0 = g(p0)
    alpha_0 = metric_0*v0

    init = torch.cat((p0, alpha_0))
    F = compute_F(param_metric)
    
    t_sort = torch.sort(t)
    t = t_sort.values + torch.linspace(0.,10e-10,len(t_sort.values))
    
    #resolution
    y0 = init
    t_eval = torch.stack((t,t))

    term = to.ODETerm(F)
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)
    jit_solver = torch.compile(solver)

    sol = jit_solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))

    #réindexation
    gamma = torch.vmap(lambda k:sol.ys[0][k])(t_sort.indices)

    return gamma


def transport_par(w,t0,tf,p0,v0,N_transport):
    t = torch.linspace(t0,tf,N_transport)

import torch
import torchode as to
import matplotlib.pyplot as plt


#g(x) = [g_11(x) , g_22(x),... , g_nn(x)]
def exp_para_lin(param_metric,p0,v0,w_i,t_i):
    #print("inside exp_para")
    #print("t_i",t_i)
    #print("p0",p0)
    #print("v0",v0)
    #print("w_i",w_i)
    exp_para = torch.vmap(lambda t:p0+w_i+v0*t)(t_i)
    return exp_para

def exp_para_logistic(param_metric,p0,v0,w_i,t_i):
    exp_para = torch.vmap(lambda t:1/(1 + (1/p0 - 1)*torch.exp(-v0*t/(p0*(1-p0)))))(t_i)
    return exp_para

def exp_para_poly1D(param_metric,p0,v0,w_i,t_i):
    g = poly_metric(param_metric)
    geo = geodesic_1D(p0[0],v0[0],g,t_i)
    return geo


def compute_poly(coeffs,deg,x):
     x_pow = torch.exp(torch.arange(0,deg+1,1) * torch.log(x))
     return torch.sum(coeffs*x_pow)

def poly_metric(param_metric):
    deg =len(param_metric)-1
    def g(x):
        return 1/compute_poly(param_metric,deg,x).reshape((1,1))
    return g

def geodesic_1D(p0,v0,g,t):
    F = lambda t,y:v0 * g(y)/g(p0)
    
    #le solver prend un tableau trié sans doublons
    #on trie
    t=torch.cat((t,torch.zeros(1)))
    t_sort = torch.sort(t)
    t = t_sort.values + torch.linspace(0.,10e-10,len(t_sort.values))
    #resolution
    y0 = p0.reshape((1,1))
    t_eval = t.reshape((1,-1))

    term = to.ODETerm(F)
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)
    jit_solver = torch.compile(solver)
    
    sol = jit_solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))
    #réindexation
    tmp= torch.zeros_like(sol.ys[0].reshape((-1,)))
    gamma = tmp.scatter(0,t_sort.indices,sol.ys[0].reshape((-1,))).reshape((-1,1))[:-1]

    return gamma


p0 = torch.tensor([0.8])
v0 = torch.tensor([2.])
y0 = p0.reshape((1,1))
param_metric = torch.tensor([1.,0.,0.])
exp_para_poly1D(param_metric,p0,v0,0,torch.tensor([.1,.2,.3,.4,1.5]))



def compute_F(param_metric,dim_geo):
    g = poly_metric(param_metric)
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
    g = poly_metric(param_metric)
    dim_geo = len(p0)

    metric_0 = g(p0)
    alpha_0 = metric_0*v0

    init = torch.cat((p0, alpha_0))
    F = compute_F(param_metric,dim_geo)
    
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


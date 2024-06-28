import torch
import torchode as to
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from torch import nn

#g(x) = [g_11(x) , g_22(x),... , g_nn(x)]
def exp_para_lin(param_metric,t0,p0,v0,w_i,t_i):
    #print("inside exp_para")
    #print("t_i",t_i)
    #print("p0",p0)
    #print("v0",v0)
    #print("w_i",w_i)
    def lin(t):
        #print(t)
        return p0+w_i+v0*(t-t0)
    exp_para = torch.vmap(lin)(t_i)
    return exp_para


def exp_para_logistic(param_metric,t0,p0,v0,w_i,t_i):
    exp_para = torch.vmap(lambda t:1/(1 + (1/p0 - 1)*torch.exp(-v0*(t-t0)/(p0*(1-p0)))))(t_i)
    return exp_para

'''
ESSAIS AVEC LES SOLVEURS D'ODE

def exp_para_poly1D(param_metric,t0,p0,v0,w_i,t_i):
    g = poly_metric(param_metric)
    geo = geodesic_1D_pointwise(t0,p0,v0,g,t_i)
    return geo

def compute_poly(coeffs,deg,x):
     x_pow = torch.exp(torch.arange(0,deg+1,1) * torch.log(x))
     return torch.sum(coeffs*x_pow)

def poly_metric(param_metric):
    deg =len(param_metric)-1
    def g(x):
        return 1/compute_poly(param_metric,deg,x).reshape((1,1))
    return g

def geodesic_1D(t0,p0,v0,g,t):
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

def geodesic_1D_point(t0,y0,v0,g):
    F = lambda t,y:v0 * g(y)/g(p0)
    def f(t_eval):
       term = to.ODETerm(F)
       step_method = to.Dopri5(term=term)
       step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
       solver = to.AutoDiffAdjoint(step_method, step_size_controller)
       jit_solver = torch.compile(solver)
       sol = jit_solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))
       return sol.ys[0][1]
    return f


def eval_geodesic_1D(t0,y0,v0,g):
    F = lambda t,y:v0 * g(y)/g(p0)
    def f(t_eval):

       result = odeint(F,y0,t_eval)
       return result[1]
    return f

def  geodesic_1D_pointwise(t0,p0,v0,g,t_i):
    y0 = p0.reshape((1,1))
    f = geodesic_1D_point(t0,y0,v0,g)
    print("t_i",t_i)
    print("t0",t0)
    n_i = len(t_i)
    t_eval = torch.vmap(lambda t:torch.linspace(t0,t,2))(t_i)
    print("t_eval",t_eval)

    geo = torch.zeros((n_i,1))
    for i in range(n_i):
        t_eval = torch.tensor([[t0,t_i[i]]])
        geo[i] = f(t_eval)
    return geo

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

'''


class NeuralNetwork(nn.Module):
    def __init__(self,dim_geo,nb_neurones):

        self.nb_neurones = nb_neurones
        self.dim_geo = dim_geo
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(dim_geo, nb_neurones),
            nn.Sigmoid(),
            nn.Linear(nb_neurones, nb_neurones),
            nn.Sigmoid(),
            nn.Linear(nb_neurones, nb_neurones),
            nn.Sigmoid(),
            nn.Linear(nb_neurones, dim_geo),
        )

    def init_id(self):
        #couche 1
        nn.init.zeros_(self.linear_relu_stack[0].weight)
        with torch.no_grad():
            for i in range(self.dim_geo):
                self.linear_relu_stack[0].weight[i, i] = 1
        nn.init.zeros_(self.linear_relu_stack[0].bias)
        
        #couche 2
        nn.init.eye_(self.linear_relu_stack[2].weight)
        nn.init.zeros_(self.linear_relu_stack[2].bias)

        nn.init.eye_(self.linear_relu_stack[4].weight)
        nn.init.zeros_(self.linear_relu_stack[4].bias)

        #couche 3
        nn.init.zeros_(self.linear_relu_stack[6].weight)
        with torch.no_grad():
            for i in range(self.dim_geo):
                self.linear_relu_stack[6].weight[i, i] = 1
        nn.init.zeros_(self.linear_relu_stack[6].bias)    

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def extrait_param(self):
        parameters_tensor = torch.cat([param.view(-1) for param in self.parameters()])
        return parameters_tensor
    
    def import_param(self,new_param):
        start = 0
        new_parameters_list = []
        for param in self.parameters():
            end = start + param.numel()  # Nombre total d'éléments dans ce paramètre
            new_parameters_list.append(new_param[start:end].view(param.size()))  # Reshape et ajout à la liste
            start = end

        for param, new_param in zip(self.parameters(), new_parameters_list):
            param.data.copy_(new_param)

    def get_grad(self):
        grad_param = []
        for param in self.parameters():
           if param.requires_grad:
               grad_param.append(param.grad.reshape(-1))
        return torch.cat(grad_param)


def neural_exp_para_func(dim_geo,nb_neurones):
    network = NeuralNetwork(dim_geo,nb_neurones)
    network.init_id()
    def exp_para_net(param_metric,t0,p0,v0,w_i,t_i):
        #print(network.extrait_param())
        network.import_param(param_metric)
        para_lin = exp_para_lin(param_metric,t0,p0,v0,w_i,t_i)
        para_net = network.forward(para_lin)
        return para_net
    return exp_para_net,network
    

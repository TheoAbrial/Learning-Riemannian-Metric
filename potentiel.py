import torch

#theta (t0m p0m v0m betam log_var_xi log_var_tau log_var_epsilon param_metric)

#z_pop (t0 p0 v0 beta)
#z_ind_i (tau_i xi_i s_i)
#z = (z_pop z_1 z_2 ... z_n)


#beta -> (dim_geo-1,dim_pca)
#s_i -> (1,dim_pca)



def minus_log_prior_ind(z_ind,theta):
    z_ind_squarred=z_ind**2 #tau_i^2, xi_i^2, s^2
    return torch.sum(z_ind_squarred[:2] / (2*torch.exp(theta[3:5]))) + torch.sum(z_ind_squarred[2:]) + torch.sum(theta[3:5]/2)

# var_pop (var_t0 var_p0 var_v0 var_beta)
def minus_log_prior_pop(z_pop,theta_pop,var_pop,dim_geo):
    z_pop_minus_theta_squarred = (z_pop-theta_pop)**2
    zpmts = z_pop_minus_theta_squarred
    return ((zpmts[0]/(2*var_pop[0])) + torch.log(var_pop[0])/2) + torch.sum((zpmts[1:1+dim_geo] / (2*var_pop[1])) + torch.log(var_pop[1])/2) + torch.sum((zpmts[1+dim_geo:1+2*dim_geo] / (2*var_pop[2])) +torch.log(var_pop[2])/2) + torch.sum((zpmts[1+2*dim_geo:]) / (2*var_pop[3]) + torch.log(var_pop[3])/2)

def minus_log_prior(z_rec,z_pop,theta,theta_pop,var_pop,dim_geo):
    prior_z_ind = torch.func.vmap(lambda z_ind:minus_log_prior_ind(z_ind,theta))(z_rec)
    return torch.sum(prior_z_ind) + minus_log_prior_pop(z_pop,theta_pop,var_pop,dim_geo)


def potentiel(exp_para,minus_log_prior,time,y,mask,var_param,dim_geo:int,dim_pca:int,nb_max_measures:int):
    
    def U(z_line,theta):   
        t0 = z_line[0]
        p0 = z_line[1:1+dim_geo]
        v0 = z_line[1+dim_geo : 1+2*dim_geo]
        beta = torch.reshape(z_line[1 + 2*dim_geo : 1 + 2*dim_geo + (dim_geo-1)*dim_pca],(dim_geo-1,dim_pca))
        z_pop = z_line[:1 + 2*dim_geo + (dim_geo-1)*dim_pca]
        z_rec= torch.reshape(z_line[1 + 2*dim_geo + (dim_geo-1)*dim_pca:],(-1,2+dim_pca))   

        theta_pop = theta[:1 + 2*dim_geo + (dim_geo-1)*dim_pca]
        theta_var = torch.exp(theta[1 + 2*dim_geo + (dim_geo-1)*dim_pca : 4 + 2*dim_geo + (dim_geo-1)*dim_pca])
        param_metric = theta[4 + 2*dim_geo + (dim_geo-1)*dim_pca:]

        minus_log_prior = minus_log_prior(z_rec,z_pop,theta_pop,var_param,dim_geo)

        base= derive_ortho_base_v0(v0,dim_geo) 
        w = compute_w_from_s(z_rec,beta,base)

        evaluation = torch.func.vmap(lambda t_i,w_i:exp_para(param_metric,t0,p0,v0,w_i,t_i))(w,time)
        minus_log_posterior = torch.sum(((y-evaluation)**2)/(2*torch.exp(theta_var[2])) + theta_var[2]/2)

        return minus_log_posterior + minus_log_prior(z_rec,theta)
    
    return U

#TODO
def derive_ortho_base_v0(v_0,dim_geo:int):
    return 0

def compute_w_from_s(z_rec,beta,base):
    z_s=z_rec[:,2:]
    beta_ortho=torch.linalg.matmul(base,beta)# (dim_geo,dim_rud)
    return torch.func.vmap(lambda x:torch.linalg.matmul(beta_ortho,x))(z_s)

def time_param_ind(t_ind,z_ind,t0):

    return torch.exp(z_ind[1]) * (t_ind - z_ind[0] - t0) + t0

def time_param(time,mask,z_rec,t_0):
    
    time_para=torch.func.vmap(lambda ti,z_ind:time_param_ind(ti,z_ind,t_0))(time,z_rec)
    return time_para*mask

import torch

#theta (t0m p0m v0m betam log_var_tau log_var_xi log_var_epsilon param_metric)

#z_pop (t0 p0 v0 beta)
#z_ind_i (tau_i xi_i s_i)
#z = (z_pop z_1 z_2 ... z_n)
# var_pop (var_t0 var_p0 var_v0 var_beta)

#beta -> (dim_geo-1,dim_pca)
#s_i -> (1,dim_pca)

def minus_log_prior_ind(z_ind,theta_var):

    z_ind_squarred=z_ind**2 #tau_i^2, xi_i^2, s^2

    return torch.sum(z_ind_squarred[:2] / (2*torch.exp(theta_var[0:2])) + theta_var[0:2]/2) + torch.sum(z_ind_squarred[2:]) 

def minus_log_prior_pop(z_pop,theta_pop,var_pop,dim_geo):
    z_pop_minus_theta_squarred = (z_pop-theta_pop)**2
    zpmts = z_pop_minus_theta_squarred
    return ((zpmts[0]/(2*var_pop[0])) + torch.log(var_pop[0])/2) + torch.sum((zpmts[1:1+dim_geo] / (2*var_pop[1])) + torch.log(var_pop[1])/2) + torch.sum((zpmts[1+dim_geo:1+2*dim_geo] / (2*var_pop[2])) +torch.log(var_pop[2])/2) + torch.sum((zpmts[1+2*dim_geo:]) / (2*var_pop[3]) + torch.log(var_pop[3])/2)

def minus_log_prior(z_rec,z_pop,theta_var,theta_pop,var_pop,dim_geo):
    prior_z_ind = torch.func.vmap(lambda z_ind:minus_log_prior_ind(z_ind,theta_var))(z_rec)
    return torch.sum(prior_z_ind) + minus_log_prior_pop(z_pop,theta_pop,var_pop,dim_geo)


def potentiel(exp_para,time,y,mask,var_param,dim_geo:int,dim_pca:int):
    nb_max_measures = len(time[0])
    nb_patients = len(time)
    
    def U(z_line,theta,tab_loss,ajoute):   
        #print("z_line",z_line)
        t0 = z_line[0]
        p0 = z_line[1:1+dim_geo]
        v0 = z_line[1+dim_geo : 1+2*dim_geo]
        beta = torch.reshape(z_line[1 + 2*dim_geo : 1 + 2*dim_geo + (dim_geo-1)*dim_pca],(dim_geo-1,dim_pca))
        z_pop = z_line[:1 + 2*dim_geo + (dim_geo-1)*dim_pca]
        z_rec= torch.reshape(z_line[1 + 2*dim_geo + (dim_geo-1)*dim_pca:],(-1,2+dim_pca))
        z_s=z_rec[:,2:]

        theta_pop = theta[:1 + 2*dim_geo + (dim_geo-1)*dim_pca]
        theta_var = theta[1 + 2*dim_geo + (dim_geo-1)*dim_pca : 4 + 2*dim_geo + (dim_geo-1)*dim_pca]
        param_metric = theta[4 + 2*dim_geo + (dim_geo-1)*dim_pca:]

        m_log_prior = minus_log_prior(z_rec,z_pop,theta_var,theta_pop,var_param,dim_geo)

        if(dim_pca > 0):
            base= derive_ortho_base_v0(v0,dim_geo) 
            w = compute_w_from_s(z_s,beta,base)
        else:
            w = torch.zeros((nb_patients,dim_geo))

        #time_param est multipliée par le masque de sorte que les valeurs manquantes sont des zéros
        altered_time = time_param(time,mask,z_rec,t0)
        #print(" y:",y)
        #print("altered_time",altered_time)
        #print("p0",p0)
        #print("w",w)
        #print("first exp_para",exp_para(param_metric,t0,p0,v0,w[0],altered_time[0]))

        vector_mask = torch.vmap(torch.vmap(lambda bool:torch.Tensor.repeat(bool,dim_geo)))(mask)
        #print("mask",vector_mask)
        evaluation = torch.vmap(lambda t_i,w_i,mask_i:mask_i*exp_para(param_metric,t0,p0,v0,w_i,t_i) + (1-mask_i)*p0)(altered_time,w,vector_mask)
        #print("\n eval",evaluation)
        
        loss =torch.sum(((y-evaluation)**2))
        if ajoute:
            tab_loss[0]+=1
            if(((int)(tab_loss[0]))%200 == 0):
                #print("eval",evaluation)
                tab_loss.append(loss)
                print("loss:",loss)

        minus_log_posterior = torch.sum(((y-evaluation)**2)/(2*torch.exp(theta_var[2])) + theta_var[2]/2)

        return minus_log_posterior + m_log_prior
    
    return U

def derive_ortho_base_v0(v0,dim_geo:int):
    e = torch.zeros(dim_geo)
    e[0] = 1
    alpha = -torch.sign(v0[0]+10e-10) * torch.norm(v0)
    u = v0 - alpha * e
    v = u / torch.norm(u)

    q_matrix = torch.eye(dim_geo) - 2*torch.outer(v,v)
    return q_matrix[:][1:]

def compute_w_from_s(z_s,beta,base):
    beta_ortho=torch.linalg.matmul(beta,base)# (dim_geo,dim_rud)
    return torch.func.vmap(lambda x:torch.linalg.matmul(x,beta_ortho))(z_s)

def time_param_ind(t_ind,z_ind,t0):

    return torch.exp(z_ind[1]) * (t_ind - z_ind[0] - t0) + t0

def time_param(time,mask,z_rec,t_0):
    #print("inside_time_param")
    #print("time",time)
    #print("mask",mask)
    #print("z_rec",z_rec)
    #print("t0",t_0)
    time_para=torch.func.vmap(lambda ti,z_ind:time_param_ind(ti,z_ind,t_0))(time,z_rec)
    return time_para*mask

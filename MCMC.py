
# -----------------------------------------------------------
# MCMC functions
# -----------------------------------------------------------

import Graph
import GGP
import numpy as np
import scipy.sparse as sps
import scipy.special as sc
import scipy.stats as scs
import time


def update_n_ABC(Graph,count ,K , w, logw, w_rem, var, alpha, sigma, tau, delta, nbABC,traffic, counted_path, paths_w_counts,ind):
        """
         ABC MCMC step
        """
    O=np.array(list(Graph.network))[ind]

    D=np.array(list(Graph.network))[ind]
    
    for i in range(nbABC):
 
        new_n = np.random.randint(-10,10)
        
        if new_n == 0:
            continue
        
        j,k = np.random.randint(0,len(ind)),np.random.randint(0,len(ind))
        l,m = ind[j], ind[k]
        new_count = count[(j,k)] + new_n
        
        
        if new_count <= 0:
            continue
    
        new_counted_path=counted_path.copy()
        
        
        o=np.array(list(Graph.network))[l]
        d=np.array(list(Graph.network))[m]

        if (o,d) in paths_w_counts.keys(): 
            for v in paths_w_counts[(o,d)]: 
                
                new_counted_path[v] = new_counted_path[v]+ count[(j,k)] - new_count
            
        else: 
            
            
            path = Graph.get_path(o,d)
            traffic_dict = counts_in_path(Graph,path)
            paths_w_counts[(o,d)] = list(traffic_dict.keys())


            if traffic_dict =={}:
                continue

            for pth in traffic_dict.keys():
                counted_path[pth] = traffic_dict[pth]
                new_counted_path[pth]= traffic_dict[pth]+ count[(j,k)] - new_count
                traffic[pth]= traffic_dict[pth] 

        delta[l,m]= sum(((new_counted_path[p]+new_count)/traffic[p])**2 for p in paths_w_counts[(o,d)])

                      
        traffic_n= sum((new_counted_path[p]/traffic[p])**2 for p in paths_w_counts[(o,d)])
        
        total= sum(abs(traffic[p]) for p in paths_w_counts[(o,d)])
        
        if total != 0 :
            
            print(delta[l,m])
            diff_n = traffic_n#
           
            rand = np.random.uniform()

            logaccept = -(diff_n/var**2)+ (delta[l,m]/var**2)+ (new_count - count[(j,k)])*np.log(w[j]*w[k]) + np.log(np.math.factorial(count[(j,k)])/ np.math.factorial(new_count))
            
            if np.log(rand)< logaccept: 
                print("update network", (diff_n/len(paths_w_counts[(o,d)]),delta[l,m]/len(paths_w_counts[(o,d)])))
                counted_path=new_counted_path
               
                count[(j,k)]= new_count
                
    N= np.sum(count,0) + np.sum(count,1)
   
    N=N.reshape(len(w))
    return(N, delta, count ,traffic, counted_path, paths_w_counts)



def update_w(w, logw, w_rem, N, L, epsilon, sigma, tau):
    """
    Update of the weights w with an hamiltonian monte carlo step

    (C) Copyright 2015 François Caron,University of Oxford

    """
    sum_w = np.sum(w)
    
    sumall_w = sum_w + w_rem
    
    logwprop = np.copy(logw)
    p = np.random.normal(size=len(w))
    
    grad1 = grad_U(N, w, w_rem, sigma, tau)
    

    pprop = np.copy(p) - epsilon* grad1/2
    
    
    for lp in range(L):
        
        logwprop = logwprop + epsilon*pprop
        if lp!=L:
            
            pprop = pprop  - epsilon * grad_U(N, np.exp(logwprop), w_rem, sigma, tau)
    
    wprop = np.exp(logwprop)

    pprop = pprop - epsilon * grad_U(N, wprop, w_rem, sigma, tau)/2
    

    sum_wprop = np.sum(wprop)
    sumall_wprop = sum_wprop + w_rem

    temp1 = - sumall_wprop**2 + sumall_w**2 + np.sum((N-sigma-1)*(logwprop - logw) )- tau * (sum_wprop - sum_w)

    logaccept = temp1 - 0.5* np.sum(pprop**2-p**2) -np.sum(logw) + np.sum(logwprop)
    
    logaccept = logaccept + np.sum(wprop**2) - np.sum(w**2)
    
    
    if np.isinf(logaccept):
        logaccept = -np.inf

    rand = np.random.uniform()
    
    if np.log(rand)<logaccept:
        w = wprop.reshape(len(w))
        logw = logwprop
    
    
    rate = np.exp(min(0, logaccept))
    if rate==np.inf: 
        rate=1
    return (w, logw, rate)


def grad_U(N, w,w_rem, sigma, tau):

    out = - (N - sigma) + w*(tau+2*np.sum(w)+2*w_rem)- 2*w**2
    return out

def update_hyper(w, logw, w_rem, alpha, logalpha, sigma,tau, nbMH,
                  rw_std, estimate_alpha, estimate_sigma, estimate_tau,
                    hyper_alpha, hyper_sigma, hyper_tau, rw_alpha):

    """
    Update of the hyperparameters with a Metropolis Hastings step

    (C) Copyright 2015 François Caron,University of Oxford

    """
    K = len(w)   
    for nn in range(nbMH):
        sum_w = np.sum(w)
        sumall_w = sum_w + w_rem
        
        if estimate_sigma:
            sigmaprop = 1-np.exp(np.log(1-sigma) + rw_std[0]*np.random.normal())
        else:
            sigmaprop = sigma

        if estimate_tau:
            tauprop = np.exp(np.log(tau) + rw_std[1]*np.random.normal())
        else:
            tauprop = tau

        if sigmaprop>-1:
            if estimate_alpha:
                if not rw_alpha: # gamma proposal
                    logalphaprop = np.log(np.random.gamma(K, 1/( GGPpsi(2*sum_w + 2*w_rem, 1, sigmaprop, tauprop) )))
                
                else:
                    logalphaprop = logalpha+ .02*np.random.normal()

                alphaprop=np.exp(logalphaprop)
            else:
                alphaprop = alpha
                logalphaprop = logalpha


            wprop_rem = GGPsumrnd(alphaprop, sigmaprop, tauprop + 2*sum_w + 2*w_rem)
            
        else: 
            if estimate_alpha:
                if not rw_alpha: 
                    alpha2prop = np.random.gamma(K, 1/( GGPpsi((2*sum_w + 2*w_rem)/tauprop, 1, sigmaprop, 1) ))
                    logalphaprop = np.log(alpha2prop) - sigmaprop*np.log(tauprop)               
                else: 
                    logalphaprop = logalpha + 0.02*np.random.normal()                

                alphaprop = np.exp(logalphaprop)
                rate_K = np.exp( logalphaprop - np.log(-sigmaprop) + sigmaprop*np.log(tauprop + 2*sum_w + 2*w_rem ) )
                num_clust = np.random.poisson(rate_K)
                wprop_rem = np.random.gamma(-sigmaprop* num_clust, 1/(tauprop+ 2*sum_w + 2*w_rem))
                
            else:
                alphaprop = alpha
                logalphaprop = logalpha
                wprop_rem = GGPsumrnd(alphaprop, sigmaprop, tauprop + 2*sum_w + 2*w_rem)


        sum_wprop = np.sum(w)
        sumall_wprop = sum_wprop + wprop_rem

        temp1 =  (sigma - sigmaprop) * np.sum(logw) - \
                sumall_wprop**2 + sumall_w**2 - \
                (tauprop - tau - 2*wprop_rem + 2*w_rem) * sum_w
        
        
        temp2 =   K* (sc.gammaln(1-sigma) - sc.gammaln(1-sigmaprop))
        
        
        logaccept = temp1 + temp2 
        
        if estimate_alpha:
            if not rw_alpha:
                logaccept = logaccept + \
                            K * (np.log(GGPpsi((2*sum_wprop + 2*wprop_rem)/tau, 1, sigma, 1) ) + sigma*np.log(tau)- \
                            np.log(GGPpsi((2*sum_w + 2*w_rem)/tauprop, 1, sigmaprop, 1) ) - sigmaprop*np.log(tauprop) )
            else:
                logaccept = logaccept - \
                            np.exp(logalphaprop + sigmaprop*np.log(tauprop))* GGPpsi((2*sum_w + 2*w_rem)/tauprop, 1, sigmaprop, 1) + \
                            np.exp(logalpha + sigma*np.log(tau)) * GGPpsi((2*sum_wprop + 2*wprop_rem)/tau, 1, sigma, 1)+ \
                            K*(logalphaprop - logalpha)


            if hyper_alpha[0]>0:
                logaccept = logaccept + hyper_alpha[0]*( logalphaprop - logalpha)

            if hyper_alpha[1]>0:
                logaccept = logaccept - hyper_alpha[1] * (alphaprop - alpha)

        else:
            logaccept = logaccept - \
                        GGPpsi(2*sum_w + 2*w_rem, alphaprop, sigmaprop, tauprop) + \
                        GGPpsi(2*sum_wprop + 2*wprop_rem, alpha, sigma, tau)

        if estimate_tau:
            logaccept = logaccept + \
                        hyper_tau[0]*( np.log(tauprop) - np.log(tau)) - hyper_tau[1] * (tauprop - tau)

        if estimate_sigma:
            logaccept = logaccept + \
                        hyper_sigma[0]*( np.log(1 - sigmaprop) - np.log(1-sigma))- \
                        hyper_sigma[1] * (1 - sigmaprop - 1 + sigma)

        if np.isinf(logaccept):
            logaccept=-np.inf
        
        if np.log(np.random.uniform())<logaccept:
            w_rem = wprop_rem
            alpha = alphaprop
            logalpha = logalphaprop
            sigma = sigmaprop
            tau = tauprop

    
    rate2 = np.exp(min(0, logaccept))
    if rate2==np.inf: 
        rate2=1
    return(w_rem, alpha, logalpha, sigma, tau, rate2)
                           


def GGPgraphmcmc(Graph, modelparam, mcmcparam, verbose=True):

    """
    MCMC sampler 
    example of initial parameters 
    mcmcparam={'niter': 10000, 'nburn': 0,'nchains':2, 'thin': 1, "nbABC":100,
           'store_w': True,'hyper.rw_std': [.02, .02],
           'hyper.MH_nb': 10,'leapfrog.L':10 ,'leapfrog.epsilon':1,
           'leapfrog.nadapt':5000,"num_OD_nodes": 100,'delta.ABC': 0.001}

    modelparam= {'alpha': [0, 0], 'tau' :[0, 0], 'sigma':[0, 0]}


    """
    adj=Graph.adj.toarray()
    K = np.shape(adj)[0]
    
    traffic, counted_path, paths_w_counts= {},{},{}
        
    var = mcmcparam["delta.ABC"]
    niter = mcmcparam['niter']
    nchains = mcmcparam['nchains']
    nburn = mcmcparam['nburn']
    thin = mcmcparam['thin']
    L = mcmcparam["leapfrog.L"]
    nbABC=mcmcparam["nbABC"]
    nb_nodes= mcmcparam["num_OD_nodes"]

    if Graph.indices is None: 
        ind = np.random.choice(range(K), size=nb_nodes, replace=False)
    else:
        ind = Graph.indices
        nb_nodes= len(Graph.indices)
               
               
    n_samples = int((niter-nburn)/thin)
    stats = [{} for _ in range(nchains)]
    samples=[{} for _ in range(nchains)]
    

    time1 = time.time()
    for nchain in range(nchains):
        print('-----------------------------------\n')
        print('Start MCMC for GGP graphs, chain: %d/%d \n'%(nchain+1,nchains))
        print('Number of iterations: %d \n' %niter) 
        print('-----------------------------------\n')
 
        epsilon = mcmcparam["leapfrog.epsilon"]/(K)**(1/4)
        count = np.random.randint(1,size=(len(ind),len(ind)) )
        np.fill_diagonal(count, 0)
        N = np.sum(count,0) + np.sum(count, 1)
        N = N.reshape(nb_nodes)

        w = np.random.gamma(1,1,size=nb_nodes)

        logw = np.log(w)
        
        w_rem = np.random.gamma(1,1)
        

        if len(modelparam["alpha"])==2:
            alpha = 100*np.random.uniform() 
            estimate_alpha = True
        else:
            alpha = modelparam["alpha"]
            estimate_alpha = False

        logalpha = np.log(alpha)
        if len(modelparam["sigma"])==2:
            sigma = 2*np.random.uniform() - 1
            estimate_sigma = 1
        else:
            sigma = modelparam["sigma"]
            estimate_sigma = 0

        if len(modelparam["tau"])==2: 
            tau = 10*np.random.uniform() 
            estimate_tau = 1
        else:
            tau = modelparam["tau"]
            estimate_tau = 0
        if mcmcparam["store_w"]:
        
            w_st = np.zeros((n_samples, nb_nodes))
        else:
            w_st = []
        delta = np.ones(shape=(nb_nodes,nb_nodes))
        w_rem_st = np.zeros((n_samples, 1))
        alpha_st = np.zeros((n_samples, 1))
        logalpha_st = np.zeros((n_samples, 1))
        tau_st = np.zeros((n_samples, 1))
        sigma_st = np.zeros((n_samples, 1))

        rate = np.zeros((niter, 1))
        rate2 = np.zeros((niter, 1))
        for i in range(niter):
            if verbose: 
                print('i=', i)
                print('alpha=', alpha)
                print('sigma=', sigma)
                print('tau=', tau) 

            (N,delta,count,traffic, counted_path, paths_w_counts) = update_n_ABC(Graph, count, K , w, logw, w_rem, var, alpha, sigma, tau, delta, nbABC,traffic, counted_path, paths_w_counts,ind)


            (w, logw, rate[i]) = update_w(w, logw, w_rem, N, L, epsilon, sigma, tau)

            if i<mcmcparam["leapfrog.nadapt"]: 
                epsilon = np.exp(np.log(epsilon) + 0.01*(np.mean(rate[:(i+1)]) - 0.6))


            if i%2==0: 
                rw_alpha = True
            else:
                rw_alpha = False;

            (w_rem, alpha, logalpha, sigma, tau, rate2[i]) = update_hyper(w, logw, w_rem, alpha, logalpha, sigma,
            tau, mcmcparam["hyper.MH_nb"], mcmcparam["hyper.rw_std"],
            estimate_alpha, estimate_sigma, estimate_tau, 
            modelparam["alpha"], modelparam["sigma"], modelparam["tau"], rw_alpha)

            if np.isinf(alpha):
                print("nan alpha")


            if i==10:
                time2 = time.time()
                time_end = (time2-time1) * (niter/10)*(nchains-nchain);
                hours = np.floor(time_end/3600)
                minutes = (time_end - hours*3600)/60
                print('-----------------------------------\n')
                print('Start MCMC for GGP graphs \n')
                print('Number of iterations: %d \n' %niter)         
                print('Estimated computation time: %.0f hour(s) %.0f minute(s)\n' %(hours, minutes))
                print('-----------------------------------\n')


            if (i>=nburn and (i-nburn)%thin==0):
                j = int((i-nburn)/thin)
                if mcmcparam["store_w"]:
                    w_st[j, :] = w.flatten()
                w_rem_st[j] = w_rem
                logalpha_st[j] = logalpha
                alpha_st[j] = alpha
                tau_st[j] = tau        
                sigma_st[j] = sigma 

        samples[nchain]["count"]=count
        samples[nchain]["w"] = w_st
        samples[nchain]["w_rem"] = w_rem_st
        samples[nchain]["alpha"] = alpha_st
        samples[nchain]["logalpha"] = logalpha_st
        samples[nchain]["sigma"]= sigma_st
        samples[nchain]["tau"] = tau_st
        samples[nchain]["ind"]=ind
        stats[nchain]["rate"] = rate
        stats[nchain]["rate2"] = rate2
        stats[nchain]["delta"]= delta 

    time3 = time.time()-time1
    hours = np.floor(time3/3600);
    minutes = (time3 - hours*3600)/60
    print('-----------------------------------\n')
    print('End MCMC for GGP graphs\n')
    print('Computation time: %.0f hour(s) %.0f minute(s)\n' %(hours,minutes))
    print('-----------------------------------\n')

    
    return(samples, stats)
    

                 


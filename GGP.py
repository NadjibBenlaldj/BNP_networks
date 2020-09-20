
# -----------------------------------------------------------
# GGP samplers  
# (C) Copyright 2015 François Caron,University of Oxford
# Adapted from matlab implementation: https://www.stats.ox.ac.uk/~caron/code/bnpgraph/
# -----------------------------------------------------------



import numpy as np
import scipy.sparse as sps
import scipy.special as sc
import scipy.stats as scs



def GGPsumrnd(alpha, sigma, tau):
	"""
    Samples the sum of weights of a GGP
    """
    if sigma< -1e-8:
        
        K = np.random.poisson(-alpha/sigma/tau**(-sigma))
        S = np.random.gamma(-sigma*K, 1/tau)
    elif sigma < 1e-8:

        S = np.random.gamma(alpha, 1/tau)
    elif sigma==0.5 and tau==0:
 
        lambd = 2*alpha**2
        mu = alpha/np.sqrt(tau);
        S = igaussrnd(mu, lambd, 1, 1);
    else:
       
        S = etstablernd(alpha/sigma, sigma, tau)[0][0]

    return S
                
def GGPrnd(alpha,sigma,tau, **kwargs):
	"""
	samples weights from a GGP

    """
    if sigma<-1e-8:
        print("finite")
        rate = np.exp(np.log(alpha) - np.log(-sigma) + sigma*np.log(tau))
        K = np.random.poisson(rate)
        N = np.random.gamma(-sigma, 1/tau, size= K)
        T = 0
        return N

    T=kwargs.get('T',None)
    if T==None:
        T=1e-6
    maxiter=kwargs.get('maxiter',None)
    
    if maxiter ==None: 
        sigma =max(sigma,0)
        a=5
        if T<a: 

            if sigma >0:

                lograte= np.log(alpha)  - np.log(sigma) - tau*T - sc.gammaln(1-sigma) + np.log(T**(-sigma) - a**(-sigma))
                Njumps=np.random.poisson(np.exp(lograte))
                
                log_N1=- 1/sigma * np.log(-(np.random.uniform(size=(Njumps, 1)) * (a**sigma-T**sigma)-a**sigma)/ (a*T)**sigma)
            else:
                lograte= np.log(alpha)  - tau*T - sc.gammaln(1-sigma) + np.log(np.log(a) - np.log(T))
                Njumps=np.random.poisson(np.exp(lograte))
                log_N1= np.random.uniform(size=(Njumps, 1))*(np.log(a) - np.log(T))+np.log(T)
            
            
            N1= np.exp(log_N1)
            ind1=np.log(np.random.uniform(size=(Njumps, 1))) <tau*(T-N1)
            N1=N1[ind1]
        else:
            N1=np.zeros(1)
            a=T
         
        lograte= np.log(alpha) - tau*a -(1+sigma)*np.log(a)-np.log(tau) - sc.gammaln(1-sigma) 
        Njumps=np.random.poisson(np.exp(lograte))         
        log_N2=np.log(a+ np.random.exponential(1/tau,size=(Njumps,1)))
        ind2 = np.log(np.random.uniform(size=(Njumps,1)))< -(1+sigma)*(log_N2-np.log(a))
        N2=np.exp(log_N2[ind2])
        N=np.concatenate([N1,N2])

    else:
        sigma=max(sigma,0)
        Njumps=[]
        if T==None:
            if sigma>0.1:
                Njumps = 20000 
                T = np.exp(1/sigma*(np.log(alpha) - np.log(sigma) - sc.gammaln(1-sigma) - np.log(Njumps)))
            
        else: 
            T=1e-10
            
 
        if T<=0:
            return
        if Njumps==[]:
            if sigma>1e-3:
                Njumps = np.floor(np.exp(np.log(alpha) -np.log(sigma) -sc.gamma(1-sigma)- np.log(T)*sigma))
            else:
                Njumps = np.floor(-alpha*np.log(T))

        if Njumps >1e8:
            print('Expected number of jumps = %d', Njumps)


        if maxiter ==None:
            maxiter = 1e8
        elif maxiter <0 :
            print('maxiter must be positive integer')
            return
                        
    
        N = []

        k = 0
        t=T
        count = 0
        if tau< 1e8:
            log_cst=np.log(alpha) -sc.gammaln(1-sigma)-np.log(sigma)
            msigma=-sigma
            msigmainv = -1/sigma
            for k in range(maxiter):
                log_r= np.log(-np.log(np.random.uniform()))-log_cst
                if log_r > msigma*np.log(t):
                    completed=True
                    break
                t=np.exp(msigmainc*np.log(t**msigma-np.exp(log_r)))
                N.append(t)
        else: 
            log_cst=np.log(alpha) -sc.gammaln(1-sigma)-log(sigma)
            sigmap1= 1+sigma
            tauinv=1/tau
            for i in range(maxiter): 
                log_r=np.log(-np.log(np.random.uniform()))
                log_G=log_cst-sigmap1*np.log(t)-tau*t
                if log_r > log_G:
                    completed=True
                    break
                t_new=t-tauinv*np.log(1-np.exp(log_r-log_G))
                if np.log(np.random.uniform())< sigmap1*(np.log(t)-np.log(t_new)):
                    k=k+1
                    N.append(t_new)
                t=t_new

        N = N[0:k]


        if completed ==False: 
            print('T too small - Its value lower at %f' %(T*10))
            T = T*10
            N = GGPrnd(alpha, sigma, tau,**{'T':T})
            return N
    return np.array(N)



def GGPrndgraph(alpha,sigma,tau)
	"""
	samples GGP random graph

    """
    w = GGPrnd(alpha, sigma, tau)
    
    cumsum_w = np.concatenate([np.array([0]), np.cumsum(w)])
    W_star = cumsum_w[-1]

    D_star = np.random.poisson(W_star**2)

    temp = W_star * np.random.uniform(size=(D_star, 2))

    bins = histc(temp, cumsum_w)
    ind, temp1, ib = np.unique(bins.flatten(), return_index=True, return_inverse=True) -2
    w_rem = sum(w)-sum(w[ind])    
    w = w[ind] 

    ib=np.reshape(ib,(bins.shape),order='F')
    indx=ib[:,0] 
    indy=ib[:,1] 

    G= sps.csr_matrix((np.ones(len(bins[:,0])),(indx,indy)),shape=(len(ind),len(ind))).toarray()
    
    return(G,w,w_rem)


 #####################################################
 ######	             Subfunctions               ######
 #####################################################




def GGPpsi(t, alpha, sigma, tau):
	"""
	returns the Laplace exponent of a  GGP

    """
    if (sigma==0): 
        out = alpha * np.log( 1+ t/tau )
    else:
        out = alpha/sigma * ((t+tau)**sigma - tau**sigma)
    return out
    


def igaussrnd(mu, lambd, M, N):

    Y = scs.chi2.rvs(1, size=(M, N))

    if (M,N) == np.shape(mu):
        X1 = mu/(2*lambd)*(2*lambd + mu*Y -np.sqrt(4*lambd*mu*Y + mu**2*Y**2))
        X2 = mu**2 / X1
    else:
        X1 = mu/(2*lambd)*(2*lambd + mu*Y -np.sqrt(4*lambd*mu*Y + mu**2*Y**2))
        X2 = mu**2 / X1
    

    U = np.random.uniform(size=(M,N))
    P = mu/(mu + X1)
    C = (U < P)
    X = C * X1 + (np.ones(shape = (M, N)) - 1*C)* X2
    
    return X


def etstablernd(V0, alpha, tau, n=1):

#Check parameters
    if alpha<=0 or alpha>=1:
        print('alpha must be in ]0,1[')
        
    if tau <0:
        print('tau must be >=0')

    if V0<=0:
        print('V0 must be >0')
    

    lambda_alpha = tau**alpha * V0 


  
    gamma = lambda_alpha * alpha * (1-alpha)

    xi = 1/np.pi *((2+np.sqrt(np.pi/2)) * np.sqrt(2*gamma) + 1) 
    psi = 1/np.pi * np.exp(-gamma * np.pi**(2/8)) * (2 + np.sqrt(np.pi/2)) * np.sqrt(gamma * np.pi)
    w1 = xi * np.sqrt(np.pi/2/gamma)
    w2 = 2 * psi * np.sqrt(np.pi)
    w3 = xi * np.pi
    b = (1-alpha)/alpha

    samples = np.zeros(shape=(n, 1))
    for i in range(n):
        while 1:
            while 1:
                
                U = gen_U(w1, w2, w3, gamma)

                W = np.random.uniform()
                zeta = np.sqrt(ratio_B(U, alpha))
                z = 1/(1 - (1 + alpha*zeta/np.sqrt(gamma))**(-1/alpha))
                rho = np.pi * np.exp(-lambda_alpha * (1-zeta**(-2))) *(xi * np.exp(-gamma*U**2/2) * (1*(U>=0))*(1*(gamma>=1)) + \
                psi/np.sqrt(np.pi-U)* (1*(U>0))*(1*(U<np.pi)) +\
                xi *(1*(U>=0))*(1*(U<=np.pi))*(gamma<1))/((1 + np.sqrt(np.pi/2)) *np.sqrt(gamma)/zeta + z)
                
                if (U<np.pi) and ((W*rho) <=1):
                    break

            a = zolotarev(U, alpha)
            m = (b/a)**alpha * lambda_alpha
            delta = np.sqrt(m*alpha/a)
            a1 = delta * np.sqrt(np.pi/2)
            a2 = a1 + delta 
            a3 = z/a
            s = a1 + delta + a3 
            V_p = np.random.uniform()    
            N_p = np.random.normal()
            E_p = -np.log(np.random.uniform())
            if V_p<a1/s:
                X = m - delta*abs(N_p)
            elif V_p<a2/s:
                X = delta * np.random.uniform() + m
            else:
                X = m + delta + a3 * E_p
            
            E = -np.log(np.random.uniform())
            cond = (a*(X-m) + np.exp(1/alpha*np.log(lambda_alpha)-b*np.log(m))*((m/X)**b - 1) - N_p**2/2 * (1*(X<m)) - E_p * (1*(X>m+delta)))
            if ((X>=0) & (cond <=E)):
                break
               
        
        samples[i] = np.exp( 1/alpha* np.log(V0) -b*np.log(X))
    return samples



def gen_U(w1, w2, w3, gamma):

    V = np.random.uniform()
    W_p =  np.random.uniform()
    if gamma>=1:
        if (V < w1/(w1+w2)):
            U = abs(np.random.normal()) /np.sqrt(gamma)
        else:
            U = np.pi * (1 - W_p**2)
        
    else:
        if (V < w3/(w3 + w2)):
            U = np.pi * W_p
        else:
            U = np.pi * (1 - W_p**2)
        
    
    return U

def ratio_B(x, sigma):

    out = sinc(x) / (sinc(sigma * x))**sigma / (sinc((1-sigma)*x))**(1-sigma)
    
    return out
def sinc(x):
    out = np.sin(x)/x
    return out


def zolotarev(u, sigma):
    out = ((np.sin(sigma*u))**sigma * (np.sin((1-sigma)*u))**(1-sigma) / np.sin(u))**(1/(1-sigma))
    return out


def histc(x, binranges):
    indices  = np.searchsorted(binranges, x)
    return np.mod(indices+1, len(binranges)+1)



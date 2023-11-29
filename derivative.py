import numpy as np
from scipy.special import erfi
from zeta import *



def derivative_sum_LT(q_2_star=1.5, cutoff=9, d = np.array([0,0,0]), ML = 4,alpha = -1):
    ML = 4
    m_tilde_sq = (ML/np.pi)**2

    d_scalar = np.linalg.norm(d)

    if d_scalar:
        beta = d_scalar/np.sqrt(d_scalar**2 + 4*q_2_star + m_tilde_sq)
        gamma = 1/np.sqrt(1-beta**2)
        beta_norm = d/np.linalg.norm(d) 
    else:
        beta = 0
        gamma = 1
        beta_norm = d

    kappa = gamma*(np.sqrt(cutoff) - beta*np.sqrt(cutoff + 1/4*m_tilde_sq))
    if alpha == -1:
        alpha = np.round(10**(-(2*np.log(kappa)/np.log(10)-1.5)),9)
    else:
        alpha = alpha

     
    #create spherical shell containing the n vectors
    rng = np.arange(-int(np.sqrt(cutoff))-1, int(np.sqrt(cutoff))+2)
    res = (rng[:,np.newaxis, np.newaxis]**2+rng[np.newaxis,:,np.newaxis]**2+rng[np.newaxis,np.newaxis,:]**2)
    X,Y,Z = np.meshgrid(rng,rng,rng, indexing = 'ij')
    coords = np.stack((X,Y,Z), axis=3)
    r = coords[res<=cutoff]

    ####### parallel and perp components of r
    r_2 = np.einsum("ij,ij->i", r,r)
    r_parallel  = np.einsum("ij,j->i", r, beta_norm)
    r_perp_sq = r_2 -r_parallel**2
    omega_r = np.sqrt(r_2+m_tilde_sq/4)

    #find r in com frame
    r_star_parallel = gamma*(r_parallel-omega_r*beta)
    r_star_sq = r_star_parallel**2 + r_perp_sq    
    omega_r_star = gamma*(omega_r -beta*r_parallel)


    #Here the factor of 2/M_cm**2 is left out, but added in back later 

    x = q_2_star
    original = omega_r_star/omega_r*np.exp(-alpha*(r_star_sq-q_2_star))/(r_star_sq-q_2_star)
    
    #expressions for derivative of sum
    term_1= original*(alpha+1/(r_star_sq-q_2_star)) 
    term_2 = (2*beta)/(m_tilde_sq+4*x) * 1/omega_r \
        *np.exp(-alpha*(r_star_sq-x))\
        /(r_star_sq-x)*r_star_parallel\
        *(1- 2*omega_r_star**2*(alpha+1/(r_star_sq-x)))
    
    # here the derivative of gamma 
    '''
    original_sum = np.sum(original)
    term_3 = -2/gamma*beta**2/(m_tilde_sq+4*x)*original_sum
    '''
    sum_terms = np.sum(term_1+term_2)#+term_3
    return sum_terms/np.sqrt(4*np.pi) #, original_sum/np.sqrt(4*np.pi)*gamma

def pv_derivative_LT(q_2, alpha):
    x = q_2
    pv = np.pi**2/np.sqrt(x)\
        *erfi(np.sqrt(alpha*x))
    return pv/np.sqrt(4*np.pi)


def derivative_LT(q_2_star=1.5, cutoff=9, d = np.array([0,0,0]),alpha = -1, S = -1):
    ML  = 4
    m_tilde_sq = (ML/np.pi)**2

    d_scalar = np.linalg.norm(d)
    #do better here
    if d_scalar:
        beta = d_scalar/np.sqrt(d_scalar**2 + 4*q_2_star + m_tilde_sq)
        gamma = 1/np.sqrt(1-beta**2)
    else:
        beta = 0
        gamma = 1

    kappa = gamma*(np.sqrt(cutoff) - beta*np.sqrt(cutoff + 1/4*m_tilde_sq))
    if alpha == -1:
        alpha = np.round(10**(-(2*np.log(kappa)/np.log(10)-1.5)),9)
    else:
        alpha = alpha
    
    #derivative of sum and pv
    deriv_S = derivative_sum_LT(q_2_star, cutoff, d, ML, alpha)-pv_derivative_LT(q_2_star, alpha)
    if S == -1:
        S = zeta(q_2_star, cutoff, alpha, d)
    else:
        S = S
    deriv_gamma = -2*beta**2*gamma/(m_tilde_sq+4*q_2_star)
    return deriv_gamma*S + deriv_S*gamma



# x_0 = 10.4
# dx = 1e-9
# Xi_sq = 1*10**(4)
# d  = np.array([8,2,3])

# alpha = -1 #0.006166199792370737
# diff = derivative_LT(x_0+dx, Xi_sq, d ,  4,alpha)[1] - derivative_LT(x_0,Xi_sq,d, 4, alpha)[1]

# print(diff/dx)
# print(derivative_LT(x_0, Xi_sq, d, 4, alpha))
# d = np.array([4,0,0])
# x_0 = 0.05401685698378193
# alpha = -1
# print(derivative_LT(x_0, 1e4, d, alpha))
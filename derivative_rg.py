
import numpy as np
from scipy import integrate
from time import time
from scipy.special import erfi
from tqdm import tqdm
import matplotlib.pyplot as plt
import sympy as sp




def deriv_rg(q_2_star=1.5, d = np.array([0,0,0]), cutoff= 9 ,  alpha = 0.1):
    '''
    Outputs derivative of the zeta correctly **ONLY** for d = [0,0,0] 
    where the expression is trivial to evaluate for all s derivatives.
    Crucially, this does not include the s! factor, for convenience
    when taylor expanding.
    d ≠ 0 can be evaluated but will be some approximation of the true result

    The inputs are x, the cutoff, the derivative order s and d
    '''

    ML = 4
    m_tilde_sq = (ML/np.pi)**2
    
    d_scalar = np.linalg.norm(d)
    s = 4*q_2_star + m_tilde_sq
    E = np.sqrt(d_scalar**2 + 4*q_2_star + m_tilde_sq)
    if d_scalar:
        beta_norm = d/np.linalg.norm(d)
        beta = d_scalar/E
        gamma = 1/np.sqrt(1-beta**2)
    else:
        beta_norm = d
        beta = 0
        gamma = 1


    #create spherical shell containing the n vectors
    rng = np.arange(-int(np.sqrt(cutoff))-1, int(np.sqrt(cutoff))+2)
    res = (rng[:,np.newaxis, np.newaxis]**2+rng[np.newaxis,:,np.newaxis]**2+rng[np.newaxis,np.newaxis,:]**2)
    X,Y,Z = np.meshgrid(rng,rng,rng, indexing = 'ij')
    coords = np.stack((X,Y,Z), axis=3)
    r = coords[res<=cutoff]


    ####### Use Rummakainen and Gottlieb's formula
    r_2 = np.einsum("ij,ij->i", r,r)
    r_parallel  = np.einsum("ij,j->i", r, beta_norm)
    #use braodcasting to multiply each of the dot products by the beta unit vector
    r_perp_sq = r_2 -r_parallel**2
    r_parallel_sq = r_parallel**2
    r_parallel_sq = 1/gamma**2*(r_parallel+ 1/2 * d_scalar)**2 
    r_sq = r_parallel_sq+ r_perp_sq
    D = r_sq-q_2_star

    terms = np.exp(-alpha*D)/(D)*(alpha+1/D)*(1-4*d_scalar**2/(gamma**2*s**2)*r_parallel_sq)

    sum_terms = np.sum(terms)/np.sqrt(4*np.pi)

    deriv_gamma = -2*d_scalar**2/gamma/s**2
    sum_terms += -pv_derivative(q_2_star,alpha)*gamma + zeta_pv(q_2_star, alpha)*deriv_gamma

    return( sum_terms )






def pv_derivative(q_2, alpha):
    x = q_2
    pv = -np.pi**2/np.sqrt(x)\
        *erfi(np.sqrt(alpha*x))
    return pv/np.sqrt(4*np.pi)



def zeta_pv(q_2=1.5, alpha=1):
    pv = 2.0*(np.pi**2)*np.sqrt(q_2)\
        * erfi(np.sqrt(alpha*q_2))\
        - 2.0*np.exp(alpha*q_2)\
        * np.sqrt(np.pi**3)/np.sqrt(alpha)
    
    return pv/np.sqrt(4*np.pi)
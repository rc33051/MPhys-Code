import numpy as np
from scipy.special import erfi
from zeta import *

'''
Contains correct derivative for zeta d≠0 and will maybe be added to the main code
'''

def derivative_sum_LT(q_2_star=1.5, cutoff=9, d = np.array([0,0,0]), m_tilde_sq = (4/(np.pi))**2, alpha = -1, beta  =0 , gamma = 1):

    d_scalar = np.linalg.norm(d)
    if d_scalar:
        beta_norm = d/np.linalg.norm(d) 
    else:
        beta_norm = d
     
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

    #up to this point code equiv. to zeta_sum. But now the new part:

    x = q_2_star
    #original expression from KSS
    original = omega_r_star/omega_r*np.exp(-alpha*(r_star_sq-q_2_star))/(r_star_sq-q_2_star)
    
    #expressions for derivative of sum
    term_1= original*(alpha+1/(r_star_sq-q_2_star)) 
    term_2 = (2*beta)/(m_tilde_sq+4*x) * 1/omega_r \
        *np.exp(-alpha*(r_star_sq-x))\
        /(r_star_sq-x)*r_star_parallel\
        *(1- 2*omega_r_star**2*(alpha+1/(r_star_sq-x)))
    
    sum_terms = np.sum(term_1+term_2)
    return sum_terms/np.sqrt(4*np.pi) 

def pv_derivative_LT(q_2, alpha):
    x = q_2
    pv = np.pi**2/np.sqrt(x)\
        *erfi(np.sqrt(alpha*x))
    return pv/np.sqrt(4*np.pi)


def derivative_LT(q_2_star=1.5, cutoff=9, d = np.array([0,0,0]),alpha = -1, S = -1):
    ML  = 4
    m_tilde_sq = (ML/np.pi)**2

    d_scalar = np.linalg.norm(d)
    #find beta and gamma
    if d_scalar:
        beta = d_scalar/np.sqrt(d_scalar**2 + 4*q_2_star + m_tilde_sq)
        gamma = 1/np.sqrt(1-beta**2)
    else:
        beta = 0
        gamma = 1

    #cutoff in com
    kappa = gamma*(np.sqrt(cutoff) - beta*np.sqrt(cutoff + 1/4*m_tilde_sq))

    #here recommended alpha is calculated
    if alpha == -1:
        alpha = np.round(10**(-(2*np.log(kappa)/np.log(10)-1.5)),9)
    else:
        alpha = alpha
    
    
    #derivative of sum and pv
    deriv_S = derivative_sum_LT(q_2_star, cutoff, d, m_tilde_sq, alpha, beta, gamma)+pv_derivative_LT(q_2_star, alpha)

    #calculate value of zeta if ==-1 else use given value
    if S == -1:
        S = zeta(q_2_star, cutoff, alpha, d)
    else:
        S = S
    
    #derivatives of beta wrt x
    deriv_gamma = -2*beta**2*gamma/(m_tilde_sq+4*q_2_star)

    #using product rule
    return deriv_gamma*S + deriv_S*gamma




#The snippet below is used to check the derivative against 
#the formal definition of the derivative for dx 10^(-8)

# d = np.array([4,0,0])
# x_0 = 0.05401685698378193 #first zero of d = 4,0,0
# alpha = -1
# Xi_sq = 1e4
# dx = 1e-10
# print((zeta(x_0+dx, Xi_sq, alpha, d)-zeta(x_0, Xi_sq, alpha, d))/dx)
# print(derivative_LT(x_0, Xi_sq, d, alpha, S = -1))




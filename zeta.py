
import numpy as np
from scipy import integrate
from time import time
from scipy.special import erfi
from tqdm import tqdm
import matplotlib.pyplot as plt
import sympy as sp


'''
NOTES FOR ZETA FUNCTION

This file contains the zeta function and thus is the heart of the code. 
It is designed after the Kim, Sachrajda and Sharpe paper and uses their 
convergence scheme. 

'''


def zeta_sum(q_2_star=1.5, cutoff=9, alpha=1, d = np.array([0,0,0]), m_tilde_sq = (4/np.pi)**2, beta_scalar = 0, gamma=1):
    
    '''
    This function calculates the sum part of the zeta function.
    It does that using a spherical shell of radius Xi (cutoff) done
    using the np.mehsgrid function. They are Lorentz transformed, evaluated
    using the expression from K-S-S and summed.
    '''

    d_scalar = np.linalg.norm(d)
    #find the unit vector in the direction of d
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
    r_star_parallel = gamma*(r_parallel-omega_r*beta_scalar)
    r_star_sq = r_star_parallel**2 + r_perp_sq    
    omega_r_star = gamma*(omega_r -beta_scalar*r_parallel)

    #evaluate the summand
    terms = omega_r_star/omega_r*np.exp(-alpha*(r_star_sq-q_2_star))/(r_star_sq-q_2_star)
    #sum the terms
    np.random.shuffle(terms)
    result = np.sum(terms)/np.sqrt(4*np.pi)

    return result


def zeta_pv(q_2=1.5, alpha=1):
    '''
    Uses formula for PV
    '''
    pv = 2.0*(np.pi**2)*np.sqrt(q_2)\
        * erfi(np.sqrt(alpha*q_2))\
        - 2.0*np.exp(alpha*q_2)\
        * np.sqrt(np.pi**3)/np.sqrt(alpha)
    
    return pv/np.sqrt(4*np.pi)


def zeta(q_2_star=1.5, cutoff=9, alpha=-1, d = np.array([0,0,0]), ML = 4):
    '''
    The input arguments are q_2_star (often called x) the cutoff which is 
    Xi^2 (i.e. the square of the radius of the spherical shell) in lab-frame
    alpha which is the convergence parameter, which in case set to -1 will
    automatically pick the recommended value (see below) and d which is the
    the dimensionless momentum to the centre of mass frame.
    ML, although a variable is in this project set to the pion mass, 
    which will be roughly 4, so it is internally set.

    '''

    #setting ML to pion mass

    m_tilde_sq = (ML/np.pi)**2
    d_scalar = np.linalg.norm(d)
    x = q_2_star

    #function below finds gamma and beta
    if d_scalar:
        beta = d_scalar/np.sqrt(d_scalar**2 + 4*q_2_star + m_tilde_sq)
        gamma = 1/np.sqrt(1-beta**2)
    else:
        beta = 0
        gamma = 1


    #if cutoff = -1, pick cutoff for the user
    if cutoff == -1:
        kappa = np.sqrt(10**(1.24)/alpha**(1.023))
        if kappa**2<100*x:
            kappa = np.sqrt(100*x)

        omega_k = np.sqrt(kappa**2 + m_tilde_sq/4)
        Xi = gamma*(kappa + omega_k * beta)
        cutoff = Xi**2


    #if alpha = -1, pick alpha for the user
    if alpha == -1:
        #kappa is the of the cutoff radius in the com frame (equal to Xi if beta = 0)
        kappa = gamma*(np.sqrt(cutoff) - beta*np.sqrt(cutoff + 1/4*m_tilde_sq))
        alpha = 10**(-(2*np.log(kappa)/np.log(10)-1.5))


    #find the sum and pv terms. PV already includes a minus sign.
    sum_result  = zeta_sum(q_2_star,cutoff, alpha, d, m_tilde_sq, beta, gamma)
    #to be equivalent to the zeta function, this has to be multipled by gamma
    result = (sum_result + zeta_pv(q_2_star,alpha))*gamma
    return result


def alpha_recommended(q_2_star, cutoff, d):
    '''
    This function returns the recommended alpha value for a given 
    q_2_star, cutoff and d_scalar and thus is the duplicate of the 
    snippet in the zeta function
    '''

    d_scalar = np.linalg.norm(d)
    ML=4
    m_tilde_sq = (ML/np.pi)**2

    #Since this function is only called if d not zero, we can directly proceed
    beta = d_scalar/np.sqrt(d_scalar**2 + 4*q_2_star + 1/4*m_tilde_sq)
    gamma = 1/np.sqrt(1-beta**2)
    kappa = gamma*(np.sqrt(cutoff) - beta*np.sqrt(cutoff + 1/4*m_tilde_sq))
    recommended_alpha = 10**(-(2*np.log(kappa)/np.log(10)-1.5))

    return recommended_alpha



# def derivative(q_2_star=1.5, cutoff= 9 , s = 1, d = np.array([0,0,0])):
#     '''
#     Outputs derivative of the zeta correctly **ONLY** for d = [0,0,0] 
#     where the expression is trivial to evaluate for all s derivatives.
#     Crucially, this does not include the s! factor, for convenience
#     when taylor expanding.
#     d ≠ 0 can be evaluated but will be some approximation of the true result

#     The inputs are x, the cutoff, the derivative order s and d
#     '''

#     ML = 4
#     m_tilde_sq = (ML/np.pi)**2
#     E_cm_sq = 4*q_2_star + m_tilde_sq
    
#     d_scalar = np.linalg.norm(d)
#     #do better here
#     if d_scalar:
#         beta_norm = d/np.linalg.norm(d)
#         beta = d_scalar/np.sqrt(d_scalar**2 + 4*q_2_star + m_tilde_sq)
#         gamma = 1/np.sqrt(1-beta**2)
#     else:
#         beta_norm = d
#         beta = 0
#         gamma = 1

#     ######################
#     ###########SUM########
    
#     #create spherical shell containing the n vectors
#     rng = np.arange(-int(np.sqrt(cutoff))-1, int(np.sqrt(cutoff))+2)
#     res = (rng[:,np.newaxis, np.newaxis]**2+rng[np.newaxis,:,np.newaxis]**2+rng[np.newaxis,np.newaxis,:]**2)
#     X,Y,Z = np.meshgrid(rng,rng,rng, indexing = 'ij')
#     coords = np.stack((X,Y,Z), axis=3)
#     r = coords[res<=cutoff]


#     ####### Use Rummakainen and Gottlieb's formula
#     r_2 = np.einsum("ij,ij->i", r,r)
#     r_parallel  = np.einsum("ij,j->i", r, beta_norm)
#     #use braodcasting to multiply each of the dot products by the beta unit vector
#     r_perp_sq = r_2 -r_parallel**2
#     r_parallel_sq = r_parallel**2
#     r_parallel_sq = 1/gamma**2*(r_parallel+ 1/2 * d_scalar)**2 
#     r_sq = r_parallel_sq+ r_perp_sq
#     D = r_sq-q_2_star
#     terms = 1/(D)**(s+1)#*(1-4*beta**2/E_cm_sq*r_parallel_sq)

#     sum_terms = np.sum(terms)
#     return sum_terms/np.sqrt(4*np.pi) 



# def first_deriv(q_2_star=1.5, d = np.array([0,0,0]), cutoff= 9 ,  alpha = 0.1):
#     '''
#     Outputs derivative of the zeta correctly **ONLY** for d = [0,0,0] 
#     where the expression is trivial to evaluate for all s derivatives.
#     Crucially, this does not include the s! factor, for convenience
#     when taylor expanding.
#     d ≠ 0 can be evaluated but will be some approximation of the true result

#     The inputs are x, the cutoff, the derivative order s and d
#     '''

#     ML = 4
#     m_tilde_sq = (ML/np.pi)**2
    
#     d_scalar = np.linalg.norm(d)
#     s = 4*q_2_star + m_tilde_sq
#     E = np.sqrt(d_scalar**2 + 4*q_2_star + m_tilde_sq)
#     if d_scalar:
#         beta_norm = d/np.linalg.norm(d)
#         beta = d_scalar/E
#         gamma = 1/np.sqrt(1-beta**2)
#     else:
#         beta_norm = d
#         beta = 0
#         gamma = 1


#     #create spherical shell containing the n vectors
#     rng = np.arange(-int(np.sqrt(cutoff))-1, int(np.sqrt(cutoff))+2)
#     res = (rng[:,np.newaxis, np.newaxis]**2+rng[np.newaxis,:,np.newaxis]**2+rng[np.newaxis,np.newaxis,:]**2)
#     X,Y,Z = np.meshgrid(rng,rng,rng, indexing = 'ij')
#     coords = np.stack((X,Y,Z), axis=3)
#     r = coords[res<=cutoff]


#     ####### Use Rummakainen and Gottlieb's formula
#     r_2 = np.einsum("ij,ij->i", r,r)
#     r_parallel  = np.einsum("ij,j->i", r, beta_norm)
#     #use braodcasting to multiply each of the dot products by the beta unit vector
#     r_perp_sq = r_2 -r_parallel**2
#     r_parallel_sq = r_parallel**2
#     r_parallel_sq = 1/gamma**2*(r_parallel+ 1/2 * d_scalar)**2 
#     r_sq = r_parallel_sq+ r_perp_sq
#     D = r_sq-q_2_star
#     terms = np.exp(-alpha*D)/(D)*(alpha+1/D)*(1-4*beta**2/s*r_parallel_sq)

#     sum_terms = np.sum(terms)
#     return( sum_terms + 2*np.pi**2/np.sqrt(q_2_star)*erfi(np.sqrt(alpha*q_2_star)))/np.sqrt(4*np.pi) 


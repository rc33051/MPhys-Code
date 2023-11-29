
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


def zeta(q_2_star=1.5, cutoff=9, alpha=-1, d = np.array([0,0,0])):
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
    ML = 4
    m_tilde_sq = (ML/np.pi)**2

    d_scalar = np.linalg.norm(d)

    #function below finds gamma and beta
    if d_scalar:
        beta = d_scalar/np.sqrt(d_scalar**2 + 4*q_2_star + m_tilde_sq)
        gamma = 1/np.sqrt(1-beta**2)
    else:
        beta = 0
        gamma = 1

    #kappa is the of the cutoff radius in the com frame (equal to Xi if beta = 0)
    kappa = gamma*(np.sqrt(cutoff) - beta*np.sqrt(cutoff + 1/4*m_tilde_sq))

    #if alpha = -1, pick alpha for the user
    if alpha == -1:
        recommended_alpha = 10**(-(2*np.log(kappa)/np.log(10)-1.5))
    else:
        recommended_alpha = alpha

    #find the sum and pv terms. PV already includes a minus sign.
    sum_result  = zeta_sum(q_2_star,cutoff, recommended_alpha, d, m_tilde_sq, beta, gamma)
    #to be equivalent to the zeta function, this has to be multipled by gamma
    result = (sum_result + zeta_pv(q_2_star,recommended_alpha))*gamma
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



def derivative(q_2_star=1.5, cutoff= 9 , s = 1, d = np.array([0,0,0])):
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
    #do better here
    if d_scalar:
        beta_norm = d/np.linalg.norm(d)
        beta = d_scalar/np.sqrt(d_scalar**2 + 4*q_2_star + m_tilde_sq)
        gamma = 1/np.sqrt(1-beta**2)
    else:
        beta_norm = d
        beta = 0
        gamma = 1

    ######################
    ###########SUM########
    
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

    r_sq = 1/gamma**2*(r_parallel+ 1/2 * d_scalar)**2 + r_perp_sq
    terms = 1/(r_sq-q_2_star)**(s+1)

    sum_terms = np.sum(terms)
    return sum_terms/np.sqrt(4*np.pi) 

######THE OTHER DERIVATIVES DO NOT CONVERGE OR ARE WRONG, WILL BE REMOVED IN FUTURE ############

def first_deriv(q_2_star, cutoff,d):

    ML = 4
    m_tilde_sq = (ML/np.pi)**2
    
    d_scalar = np.linalg.norm(d)
    E_cm = np.sqrt(d_scalar**2 + 4*q_2_star + m_tilde_sq)
    #do better here
    if d_scalar:
        beta_norm = d/np.linalg.norm(d)
        beta = d_scalar/E_cm
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


    ####### LT the ns
    r_2 = np.einsum("ij,ij->i", r,r)
    r_parallel  = np.einsum("ij,j->i", r, beta_norm)
    #use braodcasting to multiply each of the dot products by the beta unit vector
    r_perp_sq = r_2 -r_parallel**2

    delta = r_parallel+ 1/2*d_scalar
    D_1 = -(1-4*(delta)**2*(d_scalar/E_cm**2)**2)

    r_sq = 1/gamma**2*(r_parallel+ 1/2 * d_scalar)**2 + r_perp_sq
    terms = -1/(r_sq-q_2_star)**(1+1)*D_1

    return np.sum(terms)/np.sqrt(4*np.pi)

def second_deriv(q_2_star, cutoff,d):

    ML = 4
    m_tilde_sq = (ML/np.pi)**2
    
    d_scalar = np.linalg.norm(d)
    E_cm = np.sqrt(d_scalar**2 + 4*q_2_star + m_tilde_sq)
    #do better here
    if d_scalar:
        beta_norm = d/np.linalg.norm(d)
        beta = d_scalar/E_cm
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


    ####### LT the ns
    r_2 = np.einsum("ij,ij->i", r,r)
    r_parallel  = np.einsum("ij,j->i", r, beta_norm)
    #use braodcasting to multiply each of the dot products by the beta unit vector
    r_perp_sq = r_2 -r_parallel**2

    delta = r_parallel+ 1/2*d_scalar
    D_1 = -(1-4*(delta**2)*(d_scalar/E_cm**2)**2)

    r_sq = 1/gamma**2*(r_parallel+ 1/2 * d_scalar)**2 + r_perp_sq
    terms1 = 2/(r_sq-q_2_star)**(1+2)*D_1**2
    D_2 = -2*4**2*(d_scalar/E_cm**3)**2*delta**2
    terms2 = -1/(r_sq-q_2_star)**(1+1)*D_2
    return np.sum(terms1+ terms2)/np.sqrt(4*np.pi)


def derivative_sympy(q_2_star, cutoff, n, d):
    #function of q_2_star (the centre of mass momentum of the particles)
    #cutoff, which is Xi^2, n which refers to the nth derivative, 
    # and d which is the relative motion wrt to the centre of mass

    ML = 4
    m_tilde_sq = (ML/np.pi)**2
    
    d_scalar = np.linalg.norm(d)
    E_cm = np.sqrt(d_scalar**2 + 4*q_2_star + m_tilde_sq)
    #do better here
    if d_scalar:
        beta_norm = d/np.linalg.norm(d)
        beta = d_scalar/E_cm
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


    ####### LT the ns
    r_2 = np.einsum("ij,ij->i", r,r)
    r_parallel  = np.einsum("ij,j->i", r, beta_norm)
    #use braodcasting to multiply each of the dot products by the beta unit vector
    r_perp_2 = r_2 -r_parallel**2
    
    

    
    x, d,m,r_par, r_perp_sq = sp.symbols('x d m r_par r_perp_sq')
    beta = d/sp.sqrt(d**2 + m**2 + 4*x)
    r_sq = (1-beta**2)*(r_par+sp.Rational(1,2)*d)**2 + r_perp_sq
    summand = 1/(r_sq-x)

    derivative_zeta = sp.diff(summand, x,n)

    function = derivative_zeta.subs({d:d_scalar, m:m_tilde_sq, x:q_2_star})
    f = sp.lambdify((r_perp_sq, r_par), function, "numpy")

    return np.sum(f(r_perp_2, r_parallel))/np.sqrt(4*np.pi)


def derivative_ks_sympy(q_2_star, d,  alpha, cutoff):
    ML = 4
    m_tilde_sq = (ML/np.pi)**2
    
    d_scalar = np.linalg.norm(d)
    E_cm = np.sqrt(d_scalar**2 + 4*q_2_star + m_tilde_sq)
    #do better here
    if d_scalar:
        beta_norm = d/np.linalg.norm(d)
        beta = d_scalar/E_cm
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


    ####### LT the ns
    r_2 = np.einsum("ij,ij->i", r,r)
    r_parallel  = np.einsum("ij,j->i", r, beta_norm)
    #use braodcasting to multiply each of the dot products by the beta unit vector
    r_perp_sq = r_2 -r_parallel**2


    ##Sympy stuff

    sp_x, sp_omega, sp_alpha, sp_omega_star, sp_r_sq, sp_gamma, sp_beta, sp_k_par, sp_k_perp, sp_d, sp_m = sp.symbols('x omega alpha omega_star r_sq gamma beta k_par k_perp d m')

    summand = (sp_omega_star/sp_omega) * sp.exp(-sp_alpha * (sp_r_sq-sp_x))/(sp_r_sq-sp_x)
    substitution_1 = sp_gamma*(sp_omega - sp_beta * sp_k_par)
    substitution_2 = sp_gamma**2*(sp_k_par - sp_beta * sp_omega)**2 + sp_k_perp**2
    substitution_3 = 1/sp.sqrt(1-sp_beta**2)
    substitution_4 = sp_d/sp.sqrt(sp_m**2 + sp_d**2 + 4*sp_x)
    summand = summand.subs(sp_omega_star, substitution_1)
    summand = summand.subs(sp_r_sq, substitution_2)
    summand = summand.subs(sp_gamma, substitution_3)
    summand = summand.subs(sp_beta, substitution_4)
    deriv_summand = sp.diff(summand, sp_x)
    pv = 2*(sp.pi**2)*sp.sqrt(sp_x)\
            * sp.erfi(sp.sqrt(sp_alpha*sp_x))\
            - 2*sp.exp(sp_alpha*sp_x)\
            * sp.sqrt(sp.pi**3)/sp.sqrt(sp_alpha)
    deriv_pv= sp.diff(pv,sp_x)
    f_pv = sp.lambdify((sp_x, sp_alpha), deriv_pv, 'scipy')

    x = q_2_star
    expr = sp.sqrt(sp_k_perp**2 + sp_k_par**2 + sp.Rational(1,4)*sp_m**2)
    deriv_summand = deriv_summand.subs(sp_omega, expr)
    deriv_summand = deriv_summand.subs([(sp_x, x), (sp_d,d_scalar), (sp_m,np.sqrt(m_tilde_sq)), (sp_alpha, alpha)])

    f = sp.lambdify(( sp_k_par,sp_k_perp), deriv_summand, 'numpy')
    return (np.sum(f(r_parallel, np.sqrt(r_perp_sq)))- f_pv(x, alpha))/np.sqrt(4*np.pi)



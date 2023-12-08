
import numpy as np
from scipy import integrate
from time import time
from scipy.special import erfi
from tqdm import tqdm
import matplotlib.pyplot as plt
import sympy as sp




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
    Xi = int(np.sqrt(cutoff))+1 #+1 to make sure xi^2 is in range
    
    max_size = 100
    partitions = int(np.ceil(Xi/max_size))
    intervals = np.arange(-partitions, partitions)*max_size #Xi appears at positive end
    #does not change the result
   
    A, B, C  = np.meshgrid(intervals,intervals, intervals) 


    result = 0
    for i in tqdm(range(len(A))):
        for j in range(len(A[i])):
            for k in range(len(A[i,j])):
                        lower_bounds = np.array([A[i,j,k],B[i,j,k],C[i,j,k]])
                        upper_bounds = lower_bounds + max_size

                        #if the minima (of magnitudes) are all below the cutoff, proceed
                        if np.sum(np.min([lower_bounds**2,upper_bounds**2], axis=0)) <=cutoff:
                            temp_result = evaluate_terms_summand(lower_bounds, upper_bounds, q_2_star, cutoff, alpha,  beta_scalar, beta_norm, gamma, m_tilde_sq)
                            result += temp_result
    return result




def evaluate_terms_summand(lower_bounds, upper_bounds, x, cutoff, alpha,  beta, beta_norm, gamma, m_tilde_sq):
        
    '''
    The function calculates part of the summand of the zeta function, depending on the
    size of the cutoff. All arguments need to be provided by external functions.
    '''

    #numbers added for labelling

    x1 = np.arange(lower_bounds[0], upper_bounds[0]) 
    y1 = np.arange(lower_bounds[1], upper_bounds[1])
    z1 = np.arange(lower_bounds[2], upper_bounds[2])

    X,Y,Z = np.meshgrid(x1,y1,z1)


    r = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    r_2 = np.einsum("ij,ij->i", r,r)


    r = r[r_2<=cutoff]
    r_2 = r_2[r_2<=cutoff]


    ###### parallel and perp components of r
    
    r_parallel  = np.einsum("ij,j->i", r, beta_norm)
    r_perp_sq = r_2 -r_parallel**2
    omega_r = np.sqrt(r_2+m_tilde_sq/4)

    #find r in com frame
    r_star_parallel = gamma*(r_parallel-omega_r*beta)
    r_star_sq = r_star_parallel**2 + r_perp_sq    
    omega_r_star = gamma*(omega_r -beta*r_parallel)

    #evaluate the summand
    terms = omega_r_star/omega_r*np.exp(-alpha*(r_star_sq-x))/(r_star_sq-x)
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



x_0  = 0.2843357183243443

x_0  = np.round(x_0, 8)


#x_0 = 1
d = np.array([2,2,2])
alpha  = -1
print(round(zeta(x_0, 1e6, alpha, d),8))
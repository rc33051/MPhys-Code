
import numpy as np
from scipy import integrate
from time import time
from scipy.special import erfi
from tqdm import tqdm
import matplotlib.pyplot as plt




def asymptotes(d = np.array([0,0,0]), cutoff=1e3, ML = 4):

    m_tilde_sq = (ML/np.pi)**2



    d_scalar = np.linalg.norm(d)
    if d_scalar:
        d_norm = d/d_scalar
        beta_0 = d_scalar/np.sqrt(d_scalar**2 + m_tilde_sq)
        gamma_0 = 1/np.sqrt(1-beta_0**2)
    else:
        d_norm = d
        beta_0 = 0
        gamma_0 = 1

    kappa = gamma_0*(np.sqrt(cutoff) - beta_0*np.sqrt(cutoff + 1/4*m_tilde_sq))
    kappa_sq = kappa**2 
    # #print kappa_sq, the radius in which we have the complete set of asymptotes
    # Xi = np.sqrt(cutoff)
    # omega_xi = np.sqrt(Xi**2 + (ML/(2*np.pi))**2)
    # kappa_sq = gamma**2*(Xi - beta_d_scalar*omega_xi)**2

    # r_star_2 = r_star_2[r_star_2<=kappa_sq]
    # r_star_2 = np.round(r_star_2, 12)
    # asymptotes = np.unique(r_star_2)

    ######################
    ###########SUM########
    
    #create spherical shell containing the n vectors
    rng = np.arange(-int(np.sqrt(cutoff))-1, int(np.sqrt(cutoff))+2)
    res = (rng[:,np.newaxis, np.newaxis]**2+rng[np.newaxis,:,np.newaxis]**2+rng[np.newaxis,np.newaxis,:]**2)
    X,Y,Z = np.meshgrid(rng,rng,rng, indexing = 'ij')
    coords = np.stack((X,Y,Z), axis=3)
    r = coords[res<=cutoff]


    #find perp and parallel components of r (wrt d)
    r_2 = np.einsum("ij,ij->i", r,r)
    r_parallel  = np.einsum("ij,j->i", r, d_norm)
    r_perp_sq = r_2 -r_parallel**2

    Delta_sq = (r_parallel - 1/2*d_scalar)**2


    A = 4
    B = m_tilde_sq + d_scalar**2 - 4*(r_perp_sq+Delta_sq)
    C = - (r_perp_sq*(m_tilde_sq+d_scalar**2) +m_tilde_sq*Delta_sq)

    x_pos = (-B + np.sqrt(B**2-4*A*C))/(2*A)

    x_pos = np.round(x_pos, 12)
    x_pos = np.unique(x_pos)
    x_pos = x_pos[x_pos<=kappa_sq]
    return x_pos

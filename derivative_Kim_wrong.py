import numpy as np
from scipy import integrate
from time import time
from scipy.special import erfi
from tqdm import tqdm

import matplotlib.pyplot as plt


'''
THIS FILE IS OLD AND WILL BE REMOVED SOON
'''



def derivative_accelerated(q_2 = 1.5, y = 0, cutoff=9, alpha = 1, d = np.array([0,0,0])):
    return pv_derivative_accelerated(q_2, alpha)  + sum_derivative_accelerated(q_2, cutoff, alpha, d) + alpha*y

def sum_derivative_accelerated(q_2=1.5, cutoff=9, alpha = 1, d = np.array([0,0,0])):
    ###here the factor of s! is not included as it will be cancelled in the Taylor expansion


    #print(q_2)
    #pion mass
    m = 1
    
    #box_length
    #L = a*(100)
    ML = 4


    #do better here
    if np.linalg.norm(d):
        beta_norm = d/np.linalg.norm(d)
        beta_d_scalar = 1/np.sqrt((1+(ML**2)/(np.pi**2*np.dot(d,d))))
        gamma = 1/np.sqrt(1-beta_d_scalar**2)
        
    else:
        beta_norm = d
        beta_d_scalar = 0
        gamma = 1
    

    beta  = beta_norm*beta_d_scalar
        
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

    #omega_r is also needed for the evaulation of k* 
    omega_r = np.sqrt(r_2+(ML/(2*np.pi))**2)
    r_star_parallel = gamma*(r_parallel-omega_r*beta_d_scalar)
    #find r in the moving frame
    r_star_sq = r_star_parallel**2 + r_perp_sq
    #find omega_r_star (omega in moving frame)
    omega_r_star = gamma*(omega_r -beta_d_scalar*r_parallel)


    terms = np.exp(-alpha*(r_star_sq-q_2))/(r_star_sq-q_2)**(1+1)*omega_r_star/omega_r

    return gamma*np.sum(terms)/(np.sqrt(4*np.pi))

def pv_derivative_accelerated(q_2, alpha):
    x = q_2
    pv = 2*np.pi**2/np.sqrt(x)*erfi(np.sqrt(alpha*x))*(1/2 - alpha*x)+2*np.pi*np.sqrt(alpha)*np.exp(alpha*x)
    return pv/np.sqrt(4*np.pi)


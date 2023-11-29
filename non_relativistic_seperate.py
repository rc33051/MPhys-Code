#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 23:52:27 2023

@author: ericrechberger
"""



import numpy as np
from scipy import integrate
from time import time
from scipy.special import erfi

import matplotlib.pyplot as plt

#number of vertices (in position space)
vertices = 100

#spacing
a = 1

#norm
alpha = 0.01


def cd_sum(q_2=1.5, cutoff=9, alpha=1):
    
    #print(q_2)
    #pion mass
    m = 1
    
    #box_length
    L = a*(100)


    

    ######################
    ###########SUM########
    
    rng = np.arange(-int(np.sqrt(cutoff))-1, int(np.sqrt(cutoff))+2)

    n_2 = (rng[:,np.newaxis, np.newaxis]**2+rng[np.newaxis,:,np.newaxis]**2+rng[np.newaxis,np.newaxis,:]**2)
    n_2 = n_2[n_2<=cutoff]

    #n_2 = np.delete(n_2, np.argwhere(n_2 == q_2))
    q_2 = q_2
    r_2 = n_2 
    
    terms = np.exp(-alpha*(r_2-q_2))/( q_2-r_2)/(np.sqrt(4*np.pi))

    return np.sum(terms)



# def cp_pv(q_2=1.5, alpha=1):
#     #reshape the function to fit into the definition of the scipy cauchy weight
#     #for the principle value evaluation
#     #I = integrate.quad(fraction, 0, k_max,weight='cauchy', wvar = q_2)
    
#     b = 1e7

#     fraction = lambda r: -np.exp(alpha*(q_2-(r**2) ))/(r+np.sqrt(q_2))*4*np.pi*r**2 *1/(2*np.pi)**3
#     fraction_2 = lambda r: np.exp(alpha*(q_2-(r**2) ))/(q_2-r**2)*4*np.pi*r**2 *1/(2*np.pi)**3

#     I_1 = integrate.quad(fraction, 0, 100, weight='cauchy', wvar =np.sqrt(q_2))[0]
#     I_2 = integrate.quad(fraction_2, 100, b)[0]
    
#     #print(I_1+I_2)
#     I = (I_1+I_2)*(2*np.pi)**3/np.sqrt(4*np.pi)

    
#     #print(I)
#     return I

def cd_pv(q_2=1.5, alpha=1):
    ttmp = 2.0*(np.pi**2)*np.sqrt(q_2)\
        * erfi(np.sqrt(alpha*q_2))\
        - 2.0*np.exp(alpha*q_2)\
        * np.sqrt(np.pi**3)/np.sqrt(alpha)
    
    return ttmp/np.sqrt(4*np.pi)


def cd_com(q_2=1.5, cutoff=9, alpha=1):
    return -cd_sum(q_2,cutoff, alpha) + cd_pv(q_2,alpha)




##Derivative of zeta function

def zeta_deriv(q_2=1.5, cutoff=9, s=1):
    ###here the factor of s! is not included as it will be cancelled in the Taylor expansion

    a = 1
    #print(q_2)
    #pion mass
    m = 1
    
    #box_length
    L = a*(100)
    

    ######################
    ###########SUM########
    

    rng = np.arange(-int(np.sqrt(cutoff))-1, int(np.sqrt(cutoff))+2)

    n_2 = (rng[:,np.newaxis, np.newaxis]**2+rng[np.newaxis,:,np.newaxis]**2+rng[np.newaxis,np.newaxis,:]**2)
    n_2 = n_2[n_2<=cutoff]

    #n_2 = np.delete(n_2, np.argwhere(n_2 == q_2))
    q_2 = q_2
    r_2 = n_2 
    terms = 1/(r_2-q_2)**(s+1)/(np.sqrt(4*np.pi))

    return np.sum(terms)



# q_2_range = np.linspace(0.01,7.01,1000)


# level_q_2 = np.argmin(q_2_range<=0.5)
# y = []
# cutoff = 10000
# for q_2 in tqdm.tqdm(q_2_range):
#     y.append(cp_com(q_2, cutoff, alpha))


# #print(y)
# y = np.array(y)
# #y[np.abs(y) > 20] = np.nan
# plt.figure(figsize=(20,10))
# plt.ylim(-50+y[level_q_2], 50+y[level_q_2])
# plt.plot(q_2_range, y)
# plt.xlabel("q^2")
# plt.ylabel("Result of c^P in COM Frame")
# plt.title("c^P for cutoff^2 = " + str(cutoff) + ", alpha = "+ str(alpha)+", and varied q^2")
# plt.axhline(0, color='0.5', linestyle='--')
# plt.show()












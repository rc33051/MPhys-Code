
from tqdm import tqdm
import numpy as np
#from standard_imports import *
import os 
from derivative import g
from derivative import Integrals as U
from gab_large_cutoff import g_ab_parallel as g_large

import matplotlib.pyplot as plt


def limiting(a,b, d_vec, x, alpha, cutoff, ML):
    d = np.linalg.norm(d_vec)
    mtilde_sq = (ML/np.pi)**2
    beta = d/np.sqrt(d**2 + 4*x+mtilde_sq)
    gam =1/np.sqrt(1-beta**2)
    print(a, b)
    #if cutoff>5e3:
    #print('ok')
    #print(a,b)
    #print(gam**(2*b+1))
    return g_large(a,b,d_vec, x, cutoff, alpha, ML )-gam**(1+2*b)*U(a,b,x,alpha)

    return g(a,b,d_vec, x, cutoff, alpha, ML )-gam**(2*b+1)*U(a,b,x,alpha)



def main():

    d_vec = np.array([1,0,0])
    x = 0.643941
    alpha = 0.1
    #cutoff = 5e4
    ML = 4

    a = 1
    b = 1

    resolution = 10
    alpha = np.logspace(-1,-7,resolution)
    cutoff = np.logspace(1,7, resolution)

    A, C = np.meshgrid(alpha, cutoff, indexing='ij')
    Z = np.zeros_like(A)
    for i in tqdm(range(len(A))):
        for j in range(len(C)):
            Z[i,j] = limiting(a,b, d_vec, x, A[i,j],C[i,j],ML)
            #derivative(1, d_vec, x, A[i,j],C[i,j],ML) #limiting(a,b,d_vec, x, A[i,j], C[i,j], ML)



    #extent = (np.log10(np.min(A)), np.log10(np.max(A)), np.log10(np.min(C)), np.log10(np.max(C)))

    #plt.imshow(np.log10(102.83955351102952-Z), extent=extent, origin='lower', aspect='auto')

    plt.contourf(np.log10(A), np.log10(C), Z, levels=30)
    plt.colorbar()

    plt.plot()
    plt.show()

    np.savez("temp", Z = Z)
    print(Z)

main()




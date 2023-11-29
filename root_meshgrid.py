#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 01:01:46 2023

@author: ericrechberger
"""
from non_relativistic_seperate import *
from scipy.optimize import root_scalar
from tqdm import tqdm

alpha  = np.logspace(0,-3,30)
cutoff  = np.logspace(0,3,30 )


ALPHA, CUTOFF = np.meshgrid(alpha,cutoff)

r = 0

roots = np.zeros_like(ALPHA)
for i in tqdm(range(len(ALPHA))):
    for j in range(len(ALPHA[i])):
        try:
            res = root_scalar(cp_com,args=(CUTOFF[i,j],ALPHA[i,j]),bracket = [r + .1, r + .9]).root
        except:
            res = r+1
        roots[i,j] = res
        
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_surface(np.log(CUTOFF)/np.log(10), np.log(ALPHA)/np.log(10), roots)

plt.xlabel('Cutoff')
plt.ylabel('Alpha')
plt.show()

plt.figure()
plt.pcolormesh(np.log(ALPHA)/np.log(10), np.log(CUTOFF)/np.log(10), roots)
plt.colorbar()
plt.xlabel('log(alpha)')
plt.ylabel('log(cutoff)')
plt.title('error based on cutoff and alpha')
plt.show()
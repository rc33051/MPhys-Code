import numpy as np
import matplotlib.pyplot as plt
from non_relativistic_seperate import *
from scipy.optimize import root_scalar
from tqdm import tqdm

k = np.linspace(1,5,10)
a = k-1.5

Xi = 10**k
alpha = 10**(-a)


print(Xi[-1],alpha[-1])
r = 0

# roots = np.zeros_like(Xi)
# for i in tqdm(range(len(Xi))):
#     try:
#         res = root_scalar(cp_com,args=(Xi[i],alpha[i]),bracket = [r + .1, r + .9]).root
#     except:
#         res = r+1
#     roots[i] = res



# print(roots)
# print(np.array_str(roots, precision=32, suppress_small=True))





import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import root_scalar
from zeta_speedy import z_d as z_d_speedy



a = np.load('roots_zeta/d_211/data.npz')
print(a['meta_data'])

#import all 

meta_data = a['meta_data']
zeros = a['zeros']
asymptotes = a['asymptotes']
Xi = float(meta_data[0])
alpha = float(meta_data[2])
d_str  = meta_data[4]

#convert string to array, seperate by spaces, remove square brackets
d = np.array([])

for i in d_str.split(" "):
    d = np.append(d, float(i.replace("[", "").replace("]", "")))


first_asymptotes = asymptotes[:]
q_2 = np.linspace(0.001, first_asymptotes[-1], 10000)

print(Xi, alpha, d)
print(np.sqrt(Xi))
#z_d_results = z_d_speedy(q_2, Xi, alpha, d)
print(a['q_2'])
print(a['z_d_results'])

plt.plot(a['q_2'], a['z_d_results'])
plt.ylim(-100,100)
plt.show()
#[1,1,0], [1,1,1], [2,0,0], [2,1,0], [2,1,1], [2,2,0], [2,2,1], [2,2,2], [3,0,0], [3,1,0], [3,1,1],
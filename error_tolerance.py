#import all the libraries needed for the code below
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import erfc


epsilon_1 = lambda alpha, cutoff: np.sqrt(np.pi/alpha)*(1/2-alpha)*erfc(np.sqrt(alpha*cutoff))+1/np.sqrt(cutoff)*np.exp(-alpha*cutoff)
e_1 = np.vectorize(epsilon_1)



#plot e_1 for a varying cutoff and alpha on a meshgrid

a = np.linspace(-10,10,300)
k = np.linspace(-10,10,300)
A, K = np.meshgrid(a,k)

cutoffs = 10**K
alphas = 10**(-A)

zetas = np.zeros_like(alphas)

tolarance = 1e-10
for i in tqdm(range(len(alphas))):
    for j in range(len(alphas[0])):
        error = e_1(alphas[i,j], cutoffs[i,j]) - tolarance
        if error <= 0:
            zetas[i,j] =  np.nan
        else:
            zetas[i,j] =  np.log(error)/np.log(10)

# find gradient of line where zeta = np.nan

#plot a line on top of the graph , where zeta = 1

k_max = np.argmin(zetas[0] == np.nan)

#print(zetas[-1])


plt.pcolormesh(A, K,zetas)
#find first value which is not nan in zetas[-1]


#top row in graph
p1 = [np.argwhere(np.logical_not(np.isnan(zetas[-1])))]
p2 = [np.argwhere(np.logical_not(np.isnan(zetas[:,0])))]


P2 = np.array([p1[0][0][0], -1])
P1 = np.array([0, p2[-1][-1][-1]])
a_P1 , k_P1 = a[P1[0]], k[P1[1]]
a_P2, k_P2 = a[P2[0]],  k[P2[1]]

#make gradient out of the two points
gradient = (k_P2-k_P1)/(a_P2-a_P1)

#find intercept
intercept = k_P1 - gradient*a_P1

f = lambda x: gradient*x + intercept

plt.plot(a, f(a), '-.', color='black', label="${:.2f}x + {:.2f}$".format(gradient, intercept))


# Properly label the axes and the plot
plt.xlabel("$-\log_{10}(\\alpha)$")
plt.ylabel("2 $\log_{10}(\Xi)$")
plt.title("$\log_{10}(\epsilon - \\tau)$ for $\\tau = 10^{" + f"{np.log10(tolarance):.0f}" + "}$")

# Format str(tolarance) so that it is in scientific notation
plt.grid(True)
plt.colorbar(label="$\log_{10}(\epsilon - \\tau)$")

plt.ylim(min(k), max(k))
plt.xlim(min(a), max(a))
plt.legend()
plt.show()

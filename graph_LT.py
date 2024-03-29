import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.spatial import ConvexHull
import seaborn as sns

def graph_LT(q_2_star=1.5, cutoff=9, d = np.array([0,0,0]), ML = 4):
    sns.set(style="whitegrid")
    sns.set_palette('muted')
    d_scalar = np.linalg.norm(d)


    m_tilde_sq = (ML/np.pi)**2
    #do better here
    if d_scalar:
        beta_norm = d/np.linalg.norm(d)   
        beta = d_scalar/np.sqrt(d_scalar**2 + 4*q_2_star + m_tilde_sq)
        gamma = 1/np.sqrt(1-beta**2)
    else:
        beta = 0
        gamma = 1
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
    r_perp = r- r_parallel[:,np.newaxis]*beta_norm

    r_perp_sq = r_2 -r_parallel**2
    omega_r = np.sqrt(r_2+m_tilde_sq/4)
    r_star_parallel = gamma*(r_parallel-omega_r*beta)
    r_star = r_star_parallel[:,np.newaxis]*beta_norm + r_perp 
    r_star_2 = np.einsum("ij,ij->i", r_star,r_star)

    #3d plot of r_star
    fig = plt.figure(figsize=(5,4), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    #set view on xy plane
    ax.view_init(elev=0, azim=90)
    #set equal aspect ratio for 3d plot

    #do using plot trisurf
    kappa = gamma*(np.sqrt(cutoff) - beta*np.sqrt(cutoff + 1/4*m_tilde_sq))
    r_star_bounded = r_star[r_star_2<=kappa**2]

    # Create Convex Hull for r_star
    hull_r_star = ConvexHull(r_star)
    hull_r = ConvexHull(r)
    hull_r_star_bounded = ConvexHull(r_star_bounded)


    # Plot the bounding surface for r_star
    for s in hull_r_star.simplices:
        s = np.append(s, s[0])  # Close the loop
        ax.plot(r_star[s, 0], r_star[s, 1], r_star[s, 2], "r-", linewidth=0.5, label = "$\\rho$")

    # Plot the bounding surface for r

    for s in hull_r.simplices:
        s = np.append(s, s[0])  # Close the loop
        ax.plot(r[s, 0], r[s, 1], r[s, 2], "b-",linewidth=0.5, label="$\Xi$")


    # Plot the bounding surface for r_star_bounded
    for s in hull_r_star_bounded.simplices:
        s = np.append(s, s[0])  # Close the loop
        ax.plot(r_star_bounded[s, 0], r_star_bounded[s, 1], r_star_bounded[s, 2], "g-", linewidth=0.5,label="$\kappa$")

    handles, labels = plt.gca().get_legend_handles_labels()
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_yticks([])

    #create suitable title
    plt.title("Bounding Surfaces for $x = {}$, $d = {}$, $\Xi = {}$, $\\kappa = {}$".format(q_2_star, d, round(np.sqrt(cutoff)), round(kappa, 2)))

    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='center right',  bbox_to_anchor=(1, 1.5))
    plt.show()

graph_LT(0.5,1e2, d=np.array([0,0,1]))




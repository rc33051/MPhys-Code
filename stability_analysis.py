import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from tqdm import tqdm
from pathlib import Path
from zeta import zeta
from zeta_large_cutoff import zeta 




def KSS(d =0, alpha = 1,x  = 0):


    kappa = np.sqrt(10**(1.24)/alpha**(1.023))
    if kappa**2<100*x:
        kappa = np.sqrt(100*x)
    ML = 4
    m_tilde_sq = ((ML)/(np.pi))**(2)

    def beta(x):
            return d/np.sqrt(m_tilde_sq + d**2+ 4*x) 
    def gamma(x):
            return 1/np.sqrt(1-beta(x)**2)

    omega_k = np.sqrt(kappa**2 + m_tilde_sq/4)

    Xi = gamma(x)*(kappa + omega_k * beta(x))

    Xi_sq = Xi**2

    return Xi_sq #-1e5




def specific_zero(d1,d2,d3, alpha, Xi, A_l,A_u,ML):
    d_vec = np.array([d1,d2,d3])
    dx = 1e-11
    lower_asy = A_l
    upper_asy = A_u
    zero = 0

    if (upper_asy-lower_asy)<dx:
        zero = (lower_asy + upper_asy)/2
    else:
        try:
            zero = root_scalar(zeta,args=(Xi, alpha, d_vec,ML),bracket = [lower_asy+dx, upper_asy-dx]).root
        except ValueError:
            print("failed at q_2 = ", lower_asy+dx, upper_asy-dx)
            print("values at these points: ", round(zeta(lower_asy+dx, Xi, alpha, d_vec, ML)), round(zeta(upper_asy-dx, Xi, alpha, d_vec,ML)))
            zero = (lower_asy + upper_asy)/2



    return zero




KSS = np.vectorize(KSS)
specific_zero = np.vectorize(specific_zero)


def stability_analysis(d_vec = np.array([0,0,0]), alpha=0.01, resolution=6, ML = 4):



    directory = "roots_zeta/ML_{}/".format(ML)
    folder_name = "d_" + str(d_vec).replace(" ", "").replace("[", "").replace("]", "")
    data = np.load(directory+folder_name+"/data.npz")


    zeros_before = data["zeros"]
    asymptotes_before = data["asymptotes"]
    lower_asy = asymptotes_before[:-1]
    upper_asy = asymptotes_before[1:]

    d_sq = np.linalg.norm(d_vec)**2
    kappas = KSS(d_sq, alpha, zeros_before)

    #for meshgrid
    alpha_min = np.log10(alpha)-1/2
    alpha_max = np.log10(alpha)+1/2
    alphas = np.logspace( alpha_min,alpha_max, resolution)
    d1, d2, d3 = d_vec[0], d_vec[1], d_vec[2]





    ALPHA_TOTAL = np.zeros((len(zeros_before), resolution, resolution))
    CUTOFF_TOTAL = np.zeros((len(zeros_before), resolution, resolution))
    Z_TOTAL = np.zeros((len(zeros_before), resolution, resolution))



    for i in tqdm(range(len(zeros_before))):#len(zeros_before)):
        ot_mins = np.log10(kappas[i])-1/2
        ot_maxs = np.log10(kappas[i])+1/2
        cutoffs = np.logspace(ot_mins, ot_maxs, resolution)
        ALPHAS, CUTOFFS = np.meshgrid(alphas, cutoffs)
        Z = np.zeros_like(ALPHAS)

        Z = specific_zero(d1,d2,d3, ALPHAS, CUTOFFS, lower_asy[i], upper_asy[i], ML)

        ALPHA_TOTAL[i] = ALPHAS
        CUTOFF_TOTAL[i] = CUTOFFS
        Z_TOTAL[i] = Z
        



    ############Saves Data#####################
    Path("root_stability").mkdir( exist_ok=True)
    directory = "root_stability/ML_{}/".format(ML)
    Path(directory).mkdir( exist_ok=True)
    folder_name = "d_" + str(d_vec).replace(" ", "").replace("[", "").replace("]", "")
    Path(directory+folder_name).mkdir( exist_ok=True)

    meta_data = np.array([alpha, resolution, str(d_vec), ML])
    zeros_before = np.array(zeros_before)
    asymptotes_before = np.array(asymptotes_before)


    np.savez(directory+ folder_name + "/data",ALPHA_TOTAL = ALPHA_TOTAL, CUTOFF_TOTAL = CUTOFF_TOTAL, Z_TOTAL = Z_TOTAL, zeros_before  = zeros_before, asymptotes_before = asymptotes_before, meta_data = meta_data)

    fig, axs = plt.subplots(5, 4, figsize=(30, 25))

    # Iterate over the columns of all_roots
    for i in range(20):
        # Calculate the row and column indices for the subplot
        row = i // 4
        col = i % 4
        
        alpha_min, alpha_max = np.log10(ALPHA_TOTAL[i].min()), np.log10(ALPHA_TOTAL[i].max())
        cutoff_min, cutoff_max = np.log10(CUTOFF_TOTAL[i].min()), np.log10(CUTOFF_TOTAL[i].max())
        extent = [alpha_min, alpha_max, cutoff_min, cutoff_max]
        # Plot each column in a separate subplot
        im = axs[row, col].imshow( np.log10(np.abs(Z_TOTAL[i]- zeros_before[i])), aspect='auto', origin='lower', extent=extent, interpolation='none' )
        axs[row, col].set_xlabel('$\log_{10}\left(\\alpha\\right)$')
        axs[row, col].set_ylabel('$\log_{10}\left(\Xi^2\\right)$')
        axs[row, col].set_title('$U_{{{}}}$  = {:.3f}'.format(i+1,zeros_before[i]))
        axs[row, col].scatter(np.log10(alpha), np.log10(kappas[i]), c = "r", s = 100, marker = '+')  # Add red point
        fig.colorbar(im,ax=axs[row, col])
    # Add colorbar


    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Show the plot
    fig.suptitle(f'Root Stability for $\\vec d  = {{{d_vec}}}$ \n Results given as $\log_{{10}}\left( \left|{{U_i-U_i(\\alpha, \Xi^2)}}\\right|\\right)$ ', fontsize=20)
    plt.savefig(directory+ folder_name + "/stability.png")



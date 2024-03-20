import numpy as np
import matplotlib.pyplot as plt
from zeta import *
from tqdm import tqdm
from pathlib import Path
from derivative import *




def file_location(d_vec = np.array([0,0,1]), ML = 4): 
    d_vec = np.array(d_vec)
    directory = "roots_zeta/ML_{}/".format(ML)

    folder_name = "d_" + str(d_vec).replace(" ", "").replace("[", "").replace("]", "")
    path = directory+ folder_name + "/data.npz"
    return path



def plot_nice(q_2= np.array([]), z_d_results = np.array([]),asymptotes = np.array([]), zeros = np.array([]),  d_vec = np.array([0,0,1]) ):
    z_d_plot = np.copy(z_d_results)
    q_2_plot = np.copy(q_2)
    first_asymptotes = np.copy(asymptotes)
    #plt.figure(figsize = (40,6))
    ###########Creates Plots#####################  


    #z_d_plot[z_d_plot<(-1e-5)] = np.nan


    plt.plot(q_2_plot, z_d_plot, linewidth = 1)
    # #insert first asymptotes with black dotted lines, thickness 1 pt
    for i in first_asymptotes:
        plt.axvline(i, linestyle = "--", color = "black", linewidth = 1, label = "Asymptotes")

    #insert zeros with red dotted lines, thickness 1 pt
    for i in zeros:
        plt.axvline(i, color = "red", linewidth = 1, label = "Zeros")

    plt.xlabel("$q^2$", ) 
    
    plt.ylabel("$Z_d$")
    #plt.title("First {} zeros of $z^d$ vs $q^2$ for $d$ = {} ". format(len(zeros), d_vec))

    plt.xlim(0,first_asymptotes[-1])

    #legend
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='best')


    plt.ylim(-50,50)
    #tickmarks

    #set x tickmarks, 100
    max_x = first_asymptotes[-1]
    steps = np.ceil(max_x/50)
    plt.xticks(np.arange(0,max_x, steps))

    plt.grid()


def KSS(d =0, alpha = 1,x  = 0): #this finds appropriate cutoff


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

KSS = np.vectorize(KSS)
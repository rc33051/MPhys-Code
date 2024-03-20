import numpy as np
import matplotlib.pyplot as plt
from zeta import *
from tqdm import tqdm
from scipy.optimize import root_scalar
from zeta_asymptotes import asymptotes as f_asymptotes
from pathlib import Path






def root_finding(d_vec = [0,0,1], alpha =0.1, ML =4):


    d_vec = np.array(d_vec)

    Xi = -1  ##so that system recommends cutoff itself


    asymptotes = f_asymptotes(d_vec, 1e4, ML)
    asymptote_max = np.max(asymptotes)


    print("Number of Asymptotes found: ",len(asymptotes))
    print("Max Asymptote: ", np.round(asymptote_max))

    #We want first 100 zeros, thus we look at the first 100 asypmtotes
    nth_root = 20
    first_asymptotes = asymptotes[:(nth_root+1)]


    missing_zero = zeta(0, Xi, alpha, d_vec)<0
    if missing_zero:
        first_asymptotes = np.append(0,first_asymptotes)
        print("missing_zero")

    dx = 1e-11
    lower_asy = first_asymptotes[:-1]
    upper_asy = first_asymptotes[1:]

    print("Finding zeros of zeta")

    zeros = np.zeros_like(lower_asy)
    for i in tqdm(range(len(lower_asy))):
        if (upper_asy[i]-lower_asy[i])<dx:
            zeros[i] = (lower_asy[i] + upper_asy[i])/2
        else:
            try:
                zeros[i] = root_scalar(zeta,args=(Xi, alpha, d_vec, ML),bracket = [lower_asy[i]+dx, upper_asy[i]-dx]).root
            except ValueError:
                print("failed at q_2 = ", lower_asy[i]+dx, upper_asy[i]-dx)
                print("values at these points: ", round(zeta(lower_asy[i]+dx, Xi, alpha, d_vec)), round(zeta(upper_asy[i]-dx, Xi, alpha, d_vec)))
                zeros[i] = (lower_asy[i] + upper_asy[i])/2



    print("Calculating function for plotting")
    q_2 = np.linspace(0, first_asymptotes[-1], 6000)
    z_d_results = np.zeros_like(q_2)
    for i in tqdm(range(len(z_d_results))):
        z_d_results[i] = zeta(q_2[i], Xi, alpha, d_vec, ML)

    z_d_plot = np.copy(z_d_results)
    q_2_plot = np.copy(q_2)




    ############Saves Data#####################
    Path("roots_zeta").mkdir( exist_ok=True)
    directory = "roots_zeta/ML_{}/".format(ML)
    Path(directory).mkdir( exist_ok=True)
    folder_name = "d_" + str(d_vec).replace(" ", "").replace("[", "").replace("]", "")
    Path(directory+folder_name).mkdir( exist_ok=True)

    #alpha and a are set internally

    if Xi == -1:
        Xi = "Set internally"
 

    meta_data = np.array([Xi, alpha, str(d_vec), ML])
    np.savez(directory+folder_name+"/data", zeros  = zeros, asymptotes = first_asymptotes, meta_data = meta_data, q_2 = q_2, z_d_results = z_d_results)


    ############Creates Plots#####################  
    for i in first_asymptotes:
        q_2_plot= np.insert(q_2_plot, np.argmax(q_2_plot >= i),i)
        z_d_plot = np.insert(z_d_plot, np.argmax(q_2_plot >= i),np.nan)

    plt.figure(figsize = (40,6))
    plt.plot(q_2_plot, z_d_plot, label = "z_d", linewidth = 1)
    #insert first asymptotes with black dotted lines, thickness 1 pt
    for i in first_asymptotes:
        plt.axvline(i, linestyle = "--", color = "black", linewidth = 1, label = "Asymptotes")

    #insert zeros with red dotted lines, thickness 1 pt
    for i in zeros:
        plt.axvline(i, color = "red", linewidth = 1, label = "Zeros")

    #label
    plt.xlabel("q^2")
    plt.ylabel("z_d")
    plt.title("First {} zeros of $z^d$ vs $q^2$ for $d$ = {} ". format(len(zeros), d_vec))

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

    plt.savefig(directory+folder_name+"/zeros_and_asymptotes_" + folder_name + ".png")
    plt.close()



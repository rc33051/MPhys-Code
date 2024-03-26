from standard_imports import *
import os


def save_unique_npz(directory, file_name, **data):
    base_path = os.path.join(directory,file_name)
    filename = f"{base_path}_{{}}.npz"
    counter = 0
    while os.path.exists(filename.format(counter)):
        counter += 1

    np.savez(filename.format(counter), **data)



def derivative_directory(d_vec, ML, n_max, alpha, cutoff):
    data = np.load(file_location(d_vec, ML))
    zeros = data['zeros']
    zeros  = np.round(zeros, 7) #take same precision as in main paper

    # cutoffs = np.zeros_like(zeros)
    # for i in range(len(zeros)):
    #     cutoffs[i] = relation_alpha_cutoff(d_vec, alpha, zeros[i], ML)

    # print(cutoffs)

    zeros = zeros[:15]

    derivative_array = np.zeros((len(zeros) , n_max))
    g_matrices = np.zeros((len(zeros), n_max+1, n_max+1))

    for i in tqdm(range(len(zeros))):
        #4e4 is more enough if we want error of 10^-8 at alpha = 0.01
        derivative_array[i], g_matrices[i] = derivative(n_max, d_vec, zeros[i], alpha, cutoff, ML )


    ############Saves Data#####################
    Path("derivatives").mkdir( exist_ok=True)
    directory = "derivatives/ML_{}/".format(ML)
    Path(directory).mkdir( exist_ok=True)
    folder_name = "d_" + str(d_vec).replace(" ", "").replace("[", "").replace("]", "")
    Path(directory+folder_name).mkdir( exist_ok=True)

    metadata = {"d_vec": d_vec, "ML": ML, "n_max": n_max, "cutoffs": cutoff,  "alpha": alpha}

    save_unique_npz(directory + folder_name,  "data", dervatives = derivative_array, gs =  g_matrices,  metadata = metadata, zeros = zeros)

    return derivative_array, metadata, zeros








def main():
    ML = 4
    print(ML)
    alpha = 1e-4
    cutoff = 5e5
    n_max = 4

    alpha = 1e-3

    #print(os.getcwd())
    # chagne working directory to the one with the data
    #os.chdir('MPhys-Code')
    #print(    np.load(file_location([0,0,0], ML)) )

    # ds = np.array([[0,0,0], [1,0,0],[1,1,0] ,[1,1,1], [2,0,0] ])
    # #ds = np.array([ [0,0,1],[0,1,1] ,[1,1,1], [0,0,2] ])
    # for i in range(len(ds)):
    #     d_vec = ds[i]
    #     print('Next ', ds[i])
    #     derivative_directory(d_vec, ML, n_max, alpha,cutoff)

    ML = 4
    print(ML)


    ds = np.array([ [0,0,0]])  #,[0,1,1] ,[1,1,1], [0,0,2] ])

    for i in range(len(ds)):
        d_vec = ds[i]
        print('Next ', ds[i])
        derivative_directory(d_vec, ML, n_max, alpha, cutoff)





def main1():

    d_vec = np.array([0,0,0])
    ML = 4
    alpha = 0.01
    n_max = 1

    data = np.load(file_location(d_vec, ML))

    zeta_d = data["z_d_results"]
    asymptotes = data["asymptotes"]
    zeros = data["zeros"]
    q_2 = data["q_2"]

    q_derivatives = np.linspace(0,5, 500)

    accurate_deriv = np.zeros_like(q_derivatives)
    #numerical derivative of zeta

    dx = np.diff(q_2)[0]   
    dy_dx = np.diff(zeta_d)/dx

    cutoffs = np.zeros_like(q_derivatives)

    for i in tqdm(range(len(q_derivatives))):
        cutoffs[i] = relation_alpha_cutoff(d_vec, alpha, q_derivatives[i], ML)

    print(cutoffs)

 

    for i in tqdm(range(len(q_derivatives))):
        #4e4 is more enough if we want error of 10^-8 at alpha = 0.01
        accurate_deriv[i] = derivative(n_max, d_vec, q_derivatives[i], alpha, cutoffs[i], ML )


    plt.figure(figsize = (40,6)) 
    plt.plot(q_2[:-1]+dx/2, dy_dx, linewidth = 1, label = 'graphical')
    plot_nice(q_derivatives,accurate_deriv, asymptotes, zeros, d_vec )


    plt.show()


def main2():


    ML =4 
    d_vec = np.array([0,0,0])   
    alpha = 1e-3

    n_max = 2


    f = lambda alpha, gamma: (30/alpha)*gamma**2




    xs = np.linspace(0.01, 9, 500)
    firsts = np.zeros((len(xs),n_max))
    for i in tqdm(range(len(xs))):
        gamma = np.sqrt(1 + np.linalg.norm(d_vec)**2/( (ML/np.pi**2 + 4 * xs[i])))
        firsts[i] = derivative(n_max, d_vec, xs[i],alpha, f(alpha, gamma) )[0]



    np.save('derivative_graph_000.npy', firsts)


    ML =4 
    d_vec = np.array([0,0,1])   
    alpha = 1e-3

    n_max = 2


    f = lambda alpha, gamma: (30/alpha)*gamma**2




    xs = np.linspace(0.01, 6.015779500086, 500)
    firsts = np.zeros((len(xs),n_max))
    for i in tqdm(range(len(xs))):
        gamma = np.sqrt(1 + np.linalg.norm(d_vec)**2/( (ML/np.pi**2 + 4  *xs[i])))
        firsts[i] = derivative(n_max, d_vec, xs[i],alpha, f(alpha, gamma) )[0]


    np.save('derivative_graph_001.npy', firsts)

    

main2()

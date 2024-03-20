from standard_imports import *
import os


def save_unique_npz(directory, file_name, **data):
    base_path = os.path.join(directory,file_name)
    filename = f"{base_path}_{{}}.npz"
    counter = 0
    while os.path.exists(filename.format(counter)):
        counter += 1

    np.savez(filename.format(counter), **data)



def derivative_directory(d_vec, ML, n_max, alpha):
    data = np.load(file_location(d_vec, ML))
    zeros = data['zeros']
    zeros  = np.round(zeros, 7) #take same precision as in main paper

    cutoffs = np.zeros_like(zeros)
    for i in range(len(zeros)):
        cutoffs[i] = relation_alpha_cutoff(d_vec, alpha, zeros[i], ML)

    print(cutoffs)

    derivative_array = np.zeros((len(zeros) , n_max))

    for i in tqdm(range(len(zeros))):
        #4e4 is more enough if we want error of 10^-8 at alpha = 0.01
        derivative_array[i] = derivative(n_max, d_vec, zeros[i], alpha, cutoffs[i], ML )


    ############Saves Data#####################
    Path("derivatives").mkdir( exist_ok=True)
    directory = "derivatives/ML_{}/".format(ML)
    Path(directory).mkdir( exist_ok=True)
    folder_name = "d_" + str(d_vec).replace(" ", "").replace("[", "").replace("]", "")
    Path(directory+folder_name).mkdir( exist_ok=True)

    metadata = {"d_vec": d_vec, "ML": ML, "n_max": n_max, "cutoffs": cutoffs,  "alpha": alpha}

    save_unique_npz(directory + folder_name,  "data", dervatives = derivative_array, metadata = metadata, zeros = zeros)

    return derivative_array, metadata, zeros








def main():
    ML = 6
    alpha = 0.01
    n_max = 3


    print(os.getcwd())
    # chagne working directory to the one with the data
    #os.chdir('MPhys-Code')
    #print(    np.load(file_location([0,0,0], ML)) )

    #ds = np.array([ [1,0,0],[1,1,0] ,[1,1,1], [2,0,0] ])
    ds = np.array([ [0,0,1],[0,1,1] ,[1,1,1], [0,0,2] ])
    for i in range(len(ds)):
        d_vec = ds[i]
        print('Next ', ds[i])
        derivative_directory(d_vec, ML, n_max, alpha)


main()
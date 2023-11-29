# MPhys-Code
This Repository is an implementation of the Lüscher Zeta function according to the Kim, Sachrajda and Sharpe. Furthermore it (taylor) expands this function about its unitary points, which are the points of infinite interaction strength, which is where the function itself is zero. This hopefully allows for faster evaluation of the function. Each file will have a short description.



## roots_zeta folder
The folder contains the first 100 roots for Z^d up to d = [4,4,4] (dimensionless momentum vector). Each file contains a data file in the format of npz (numpy zip) which can be loaded using the np.load(path+'/data.npz'), the standard loading function in python. This was chosed for convenience, while sacrificing the readability a bit, however, if desired it could be straightforwardly converted into a textfile.
The data - file contains roughly 10000 values of q_star_sq and the corresponding Z^d values, a list of the first 100 zeros (sometimes 101 if there is no asymptote at zero), a complete list of asymptotes and some metadata. Alpha in this case changes according to the convergence criterion for Z^d, while the cutoff Xi_sq remains fixed. This choice was made to ensure that the computation times do not change too much, also if the cutoff Xi_sq > 1e5 sometimes the code just fails to evaluate (it crashes). **Sidenote: This is because calculation scales as Xi_sq^(3/2) --> Xi_Sq of 1e5 is roughly 1e7.5 points, 10 MB --> crash** Hence in the metadata the value of alpha is not given, whereas Xi_sq and other choices are.

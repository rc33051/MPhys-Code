import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def g_ab(a=1,b=0,d = np.array([0,0,0]),q_2_star=1.5, cutoff=9, alpha=1,  ML = 4):
    
    '''
    This function calculates the sum part of the zeta function.
    It does that using a spherical shell of radius Xi (cutoff) done
    using the np.mehsgrid function. They are Lorentz transformed, evaluated
    using the expression from K-S-S and summed.
    '''

    m_tilde_sq = (ML/np.pi)**2
    d_scalar = np.linalg.norm(d)
    x = q_2_star

    #function below finds gamma and beta
    if d_scalar:
        beta = d_scalar/np.sqrt(d_scalar**2 + 4*x + m_tilde_sq)
        gamma = 1/np.sqrt(1-beta**2)
        beta_norm = d/np.linalg.norm(d)   
    else:
        beta = 0
        gamma = 1
        beta_norm = d


    #create spherical shell containing the n vectors
    Xi = int(np.sqrt(cutoff))+1 #+1 to make sure xi^2 is in range
    
    max_size = 50
    partitions = int(np.ceil(Xi/max_size))
    intervals = np.arange(-partitions, partitions)*max_size #Xi appears at positive end
    #does not change the result
   
    A, B, C  = np.meshgrid(intervals,intervals, intervals) 

    result = 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            for k in range(len(A[i,j])):
                        lower_bounds = np.array([A[i,j,k],B[i,j,k],C[i,j,k]])
                        upper_bounds = lower_bounds + max_size
                        args = (lower_bounds, upper_bounds, a,b, x, cutoff, alpha,  beta, beta_norm, gamma,d_scalar, m_tilde_sq)
                        #if the minima (of magnitudes) are all below the cutoff, proceed
                        if np.sum(np.min([lower_bounds**2,upper_bounds**2], axis=0)) <=cutoff:

                            temp_result = evaluate_terms_summand(*args)
                            result += temp_result
    return result


def g_ab_parallel(a=1, b=0, d=np.array([0, 0, 0]), q_2_star=1.5, cutoff=9, alpha=1, ML=4):



    m_tilde_sq = (ML/np.pi)**2
    d_scalar = np.linalg.norm(d)
    x = q_2_star

    if d_scalar:
        beta = d_scalar / np.sqrt(d_scalar**2 + 4*x + m_tilde_sq)
        gamma = 1 / np.sqrt(1 - beta**2)
        beta_norm = d / d_scalar
    else:
        beta = 0
        gamma = 1
        beta_norm = d

    Xi = int(np.sqrt(cutoff)) + 1
    max_size = 50
    partitions = int(np.ceil(Xi / max_size))
    intervals = np.arange(-partitions, partitions) * max_size
    A, B, C = np.meshgrid(intervals, intervals, intervals, indexing='ij')
    tasks = []


    for i in range(len(A)):
        for j in range(len(A[i])):
            for k in range(len(A[i, j])):
                lower_bounds = np.array([A[i, j, k], B[i, j, k], C[i, j, k]])
                upper_bounds = lower_bounds + max_size
                if np.sum(np.min([lower_bounds**2, upper_bounds**2], axis=0)) <= cutoff:
                    args = (lower_bounds, upper_bounds, a, b, x, cutoff, alpha, beta, beta_norm, gamma, d_scalar, m_tilde_sq)
                    tasks.append(args)


    with ProcessPoolExecutor() as executor:
        results = executor.map(unpack_evaluate_terms_summand, tasks)

    return sum(results)

def unpack_evaluate_terms_summand(args):
    return evaluate_terms_summand(*args)


def evaluate_terms_summand(lower_bounds, upper_bounds,a,b, x, cutoff, alpha,  beta, beta_norm, gamma,d_scalar, m_tilde_sq):
        
    '''
    The function calculates part of the summand of the zeta function, depending on the
    size of the cutoff. All arguments need to be provided by external functions.
    '''

    #numbers added for labelling

    x1 = np.arange(lower_bounds[0], upper_bounds[0]) 
    y1 = np.arange(lower_bounds[1], upper_bounds[1])
    z1 = np.arange(lower_bounds[2], upper_bounds[2])

    X,Y,Z = np.meshgrid(x1,y1,z1)


    r = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    r_2 = np.einsum("ij,ij->i", r,r)


    r = r[r_2<=cutoff]
    r_2 = r_2[r_2<=cutoff]

    r_parallel  = np.einsum("ij,j->i", r, beta_norm)
    #use braodcasting to multiply each of the dot products by the beta unit vector
    r_perp_sq = r_2 -r_parallel**2
    rho = r_parallel+ 1/2 * d_scalar
    r_parallel_star = 1/gamma*(rho)
    r_parallel_star_sq = r_parallel_star**2
    r_sq = r_parallel_star_sq+ r_perp_sq
    D = r_sq-x

    terms = np.exp(-alpha*D)/(D)**(a+1)*rho**(2*b)
    result  = np.sum(terms)
    return result

# if __name__ == "__main__":
#     a, b = 2, 1  # Example values for a and b
#     d = np.array([1, 0, 0])
#     q_2_star = 0
#     cutoff = 1e6
#     alpha = 0
#     ML = 4
#     # Call your parallel function within this protected block
#     result = g_ab_parallel(a, b, d, q_2_star, cutoff, alpha, ML)
#     print(result)



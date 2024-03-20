import numpy as np
from gab_large_cutoff import g_ab_parallel as g_ab_large
from math import comb
from math import gamma as gam_f
from scipy.special import hyp1f1


def dynamical(x = 0, d_vec = np.array([0,0,0]), ML = 4):
    d_vec = np.array(d_vec)
    d = np.linalg.norm(d_vec)
    m_tilde_sq = (ML/np.pi)**2
    beta = d/np.sqrt(d**2 + 4*x + m_tilde_sq)
    gamma = 1/np.sqrt(1-beta**2)
    return beta, gamma


def relation_alpha_cutoff(d =0, alpha = 1,x  = 0):
    ''' 
    This function finds the cutoff based on a provided alpha.
    The relation between the two was obtained by graphically 
    solving for an error of 1e-8. This was found by doing the 
    integral provided that x*100< Xi^2
    '''

    kappa = np.sqrt(10**(1.24)/alpha**(1.023))
    if kappa**2<100*x:
        kappa = np.sqrt(100*x)

    m_tilde_sq = ((ML)/(np.pi))**(2)

    def beta(x):
            return d/np.sqrt(m_tilde_sq + d**2+ 4*x) 
    def gamma(x):
            return 1/np.sqrt(1-beta(x)**2)

    omega_k = np.sqrt(kappa**2 + m_tilde_sq/4)

    Xi = gamma(x)*(kappa + omega_k * beta(x))

    Xi_sq = Xi**2

    return Xi_sq #-1e5





def lah(n,k):
    '''
    Returns the definition of the signed lah numbers
    '''
    if n == 0 and k == 0:
        return 1
    if k<1 or n<1:
        return 0
    
    return (-1)**n*comb(n-1,k-1)*gam_f(n+1)/gam_f(k+1)



def C_ij(n,i,j):
    '''
    Returns the combinatorial factors of the coefficient matrix
    '''
    if  i > n or j > n or j>i:
        return 0
    h = (i-j)
    return gam_f(h+j+1)*comb(n, h)*lah(n-h,j)


def K_ij(n = 1, i = 1,j = 0, x = 0,d = 1, ML = 4):
    '''
    Finds the full coefficients of the coefficient matrix.
    '''

    if d == 0:
        if i == n and j == 0:
            return gam_f(n+1)
        else:
            return 0
    if  i > n or j > n:
        return 0
    
    beta, gamma = dynamical(x, d, ML)
    return C_ij(n,i,j)*(2/d)**(2*(n+(j-i)))* beta**(2*(n+ 2*j-i))



def Integrals(a,b,x, alpha):
    '''
    This returns the value of the PV integral for an arbitrary value of a and b.
    Expression obtained using Mathematica; results cross-checked.
    '''
    prefactor = ((-1)**(a + b) * np.exp(x * alpha) * np.pi * alpha**(-0.5 + a - b)) / (2 * (3 + 2 * b))
    term1 = (3 + 2 * b - 2 * x * alpha) * hyp1f1(3 + a, 0.5 + a - b, -x * alpha) / gam_f(0.5 + a - b)
    term2 = x * alpha * (7 + 2 * a + 2 * b - 2 * x * alpha) * hyp1f1(3 + a, 1.5 + a - b, -x * alpha) / gam_f(1.5 + a - b)
    result = prefactor * (term1 + term2)
    angular = 4*np.pi/(2*b+1)
    return result*angular


def g(a = 1, b = 1, d = np.array([1,0,0]), x = 0 ,  cutoff = 2e4,alpha = 0.1, ML = 4):
    
    '''
    Evaluates the sum for a general a and b. 
    This uses an extension Rummakainen and Gottlieb prescription.
    '''

    # if a -b >1/2 the sum converges, so we can set alpha = 0
    if a == b+1:
        alpha = 0
    if a>b+1:
        cutoff = 1e2
        alpha = 0



    m_tilde_sq = (ML/np.pi)**2 #an internal dimensionless mass
    
    d_scalar = np.linalg.norm(d)
    s = 4*x + m_tilde_sq
    E = np.sqrt(d_scalar**2 + 4*x + m_tilde_sq)

    if d_scalar:
        beta_norm = d/np.linalg.norm(d)
        beta = d_scalar/E
        gamma = 1/np.sqrt(1-beta**2)
    else:
        beta_norm = d
        beta = 0
        gamma = 1


    #create spherical shell containing the n vectors
    rng = np.arange(-int(np.sqrt(cutoff))-1, int(np.sqrt(cutoff))+2)
    res = (rng[:,np.newaxis, np.newaxis]**2+rng[np.newaxis,:,np.newaxis]**2+rng[np.newaxis,np.newaxis,:]**2)
    X,Y,Z = np.meshgrid(rng,rng,rng, indexing = 'ij')
    coords = np.stack((X,Y,Z), axis=3)
    r = coords[res<=cutoff]


    ####### Use Rummakainen and Gottlieb's prescription
    r_2 = np.einsum("ij,ij->i", r,r)
    r_parallel  = np.einsum("ij,j->i", r, beta_norm)
    #performs the $\Gamma$ operation on the lattice
    r_perp_sq = r_2 -r_parallel**2
    rho = r_parallel+ 1/2 * d_scalar
    r_parallel_star = 1/gamma*(rho)
    r_parallel_star_sq = r_parallel_star**2
    r_sq = r_parallel_star_sq+ r_perp_sq
    D = r_sq-x

    #sums according to specified a and b
    terms_2 = np.exp(-alpha*D)/(D)**(a+1)*rho**(2*b)
    return np.sum(terms_2)#/np.sqrt(4*np.pi)



def deltaFV(a = 1, b= 0, d_vec = np.array([1,0,0]), x = 0, cutoff = 2e4, alpha = 0.01, ML = 4, cutoff_high = 1e5):
    '''
    Returns the value of the finite volume operator.
    Instead of 'brute forcing' this code tries to avoid
    unnecessary calculations by using the properties of the
    sums in the different diagonals.

    '''
    d = np.linalg.norm(d_vec)
    m_tilde_sq = (ML/np.pi)**2
    beta = np.sqrt(d**2/(4*x+m_tilde_sq+d**2))
    gam = np.sqrt(1/(1-beta**2))


    if (a==0 and b==0): #the G_00 term will always be 0
        return 0
    
    elif (a==b): #if a = b then we need the integral for convergence
        if cutoff>1e4:
            return g_ab_large(a,b,d_vec,x, cutoff, alpha, ML)-gam**(2*b+1)*Integrals(a,b,x,alpha)
        else:
            return g(a,b,d_vec,x,cutoff, alpha, ML)-gam**(2*b+1)*Integrals(a,b,x,alpha)
    
    elif (a==b+1): #if a = b+1 th sum converges but we need need higher cutoff
        if cutoff_high>1e4:
            return g_ab_large(a,b,d_vec,x, cutoff_high, 0, ML)
        else:
            return g(a,b,d_vec,x, cutoff_high, 0,   ML)
    
    elif a>b: #if a>b the sum converges quickly
        return g(a,b,d_vec,x, 1e2, 0,   ML)

    else: #if a<b the FV operator is 0
        return 0



def derivative(n_max = 1, d_vec = np.array([1,0,0]), x = 0, alpha = 0.01, cutoff = 1e4, ML = 4, cutoff_high = 1e5):

    d = np.linalg.norm(d_vec)

    FV_matrix = np.zeros((n_max+1,n_max+1))
    for i in range(n_max+1):
        for j in range(n_max+1):

            #since all coefficients will be zero if d = 0, we can skip their calculation
            if d == 0:
                if j == 0: 
                    FV_matrix[i,j] = deltaFV(i,j,d_vec, x, cutoff, alpha, ML, cutoff_high )
                else:
                    FV_matrix[i,j] = 0

            else:
                # the delta FV function has some more conditions
                FV_matrix[i,j] = deltaFV(i,j,d_vec, x, cutoff, alpha, ML, cutoff_high)

    #print(FV_matrix)

    derivatives = np.array([])
    for n in range(1,n_max+1):
        K = np.zeros((n_max+1,n_max+1))
        for i in range(n_max+1):
            for j in range(n_max+1):
                K[i,j] = K_ij(n,i,j,x,d, ML)
        D_n = K* FV_matrix
        derivatives = np.append(derivatives,np.sum(D_n ))

    return derivatives/np.sqrt(4*np.pi)
    



import numpy as np
import matplotlib.pyplot as plt
from cla_utils.exercises8 import hessenberg
from cla_utils.exercises9 import pure_QR
from q1 import triA

"""
Matrices required for Question 3
"""
def getG():
    """
    Construct the matrix required in Q3 g)
    """
    D = np.diag(np.arange(1, 16))
    O = np.ones((15,15))
    G = D + O
    return G

def getA():
    """
    Construct the matrix A required in Q3 d)
    """
    A = np.zeros((5,5))
    A[:,0] = np.arange(3,8)
    for i in range(0, 4):
        A[:,i+1] = A[:,i] + 1
    A = np.reciprocal(A)
    return A

def getB():
    """
    Create a random, symmetric, tridiagonal 5x5 matrix B
    """
    np.random.seed(3456)
    d1 = np.random.rand(5)
    d2 = np.random.rand(4)
    B = np.diag(d1) + np.diag(d2,1) + np.diag(d2,-1)
    
    return B

def getC():
    """
    Create a random, symmetric 5x5 matrix C 
    """
    np.random.seed(3456)
    C = np.random.randn(5,5)
    C = C + C.T
    return C

"""
Question 3 c)
"""
def qr_factor_tri(A):
    """
    Implements the algorithm from Q2c)
    Reduces A to R (where A = QR) via householder transformations

    :param A: an mxn-dimensional matrix
    :param kmax: an integer, the number of columns of A to reduce
    to upper triangular. If not present, will default to n.

    :return R: an mxn-dimensional matrix containing the upper
    triangular matrix
    :return V: a 2x(m-1)-dimensional matrix containing all the 2-d vectors v used
    """
    m, _ = A.shape

    # intialise V
    V = np.zeros((2, m-1))

    # k cycles from 1 to n
    for k in range(0, m):
        
        x = A[k: k+2, k]

        # initialise e_1 of length m-k+1
        e_1 = np.eye(np.size(x,0))[:,0]

        # householder algorithm (specifically for tridiagonal A)
        v_k = np.sign(x[0]) * np.linalg.norm(x) * e_1 + x
        v_k = v_k / np.linalg.norm(v_k)
        A[k: k+2, k: k+3] = A[k: k+2, k: k+3]  - 2 * np.outer(v_k, v_k) @ A[k: k+2, k: k+3]

        # Store the 2d vectors v in matrix V
        if k != m-1:
            V[:, k] = v_k
        
    return A, V


"""
Question 3 d)
"""
def qr_alg_tri(A, maxit, shift = False, return_T_array = False):
    """
    For matrix A, apply the QR algorithm till the stopping criteria and return the result.

    :param A: an mxm symmetric, tridiagonal matrix
    :param maxit: the maximum number of iterations
    :param shift: if True, apply the shifted QR algorithm
    :param return_T_array: logical

    :return Ak: the result
    :return Tarray: the array of values of Tm,m-1 at each iteration
    """
    Ak = A
    m, _ = A.shape

    # counter
    its = 0
    # list storing values of T_m,m-1 at each iteration
    Tlist = list()

    while True:
        if shift:
            # Calculate the shift mu
            a = A[m-1, m-1]
            b = A[m-1, m-2]
            d = (A[m-2,m-2] - A[m-1,m-1])/2
            mu = a - (np.sign(d) * (b**2))/(np.abs(d) + np.sqrt(d**2 + b**2))
            # Apply the shift
            Ak = Ak - mu * np.eye(m)

        # Obtain R and V from the QR factorisation algorithm
        R, V = qr_factor_tri(Ak)
        
        # Find RQ transpose via implicit multplication
        RQ = R.T
        for k in range(0, m):
            if k != m-1:
                RQ[k:k+2,:] -= 2 * np.outer(V[:,k], V[:,k]) @ RQ[k:k+2,:]
            else:
                RQ[k:k+2,:] -= 2 * RQ[k:k+2,:]
        Ak = RQ.T

        if shift:
            # Shift back again
            Ak = Ak + mu * np.eye(m)

        # Update the required variables
        its += 1
        Tlist.append(np.abs(Ak[m-1, m-2]))

        # Check stopping criteria
        if np.abs(Tlist[-1]) < 1.0e-12:
            break
        elif its+1 > maxit:
            break
    
    if return_T_array:
        return Ak, np.array(Tlist)
    else:
        return Ak

def Q3d():
    """
    Function to investigate the qr_alg_tri applied to matrix A in Q3 d)
    """
    A1 = getA()
    A2 = A1.copy()
    hessenberg(A2)
    print("Matrix obtained after applying qr_alg_tri to A:")
    print(qr_alg_tri(A2, 1000))
    print("Eigenvalues of A:")
    print(np.linalg.eig(A1)[0])
    return


"""
Question 3 e), f) and g)
"""
def Submatrix_QR_Alg(A, ApplyShift = False):
    """
    For matrix A, apply the steps outlined in Q3e) (or with a shift for Q3f))

    :param A: an mxm symmetric, real matrix
    :param shift: if True, apply the shifted QR algorithm
    """
    # reduce A to Hessenberg form
    hessenberg(A)

    m, _ = A.shape

    # create an array for the eigenvalues and for the concatenated T_arrays
    evals = np.zeros(m)
    total_T_array = np.array([])

    # loop from m to 2 backwards
    for j in range(m, 1, -1):
        # call the QR algorithm until termination
        A, T_array = qr_alg_tri(A, 5000, return_T_array = True, shift = ApplyShift)

        # update the arrays
        total_T_array = np.concatenate((total_T_array, T_array))
        evals[m-j] = A[j-1, j-1]

        # update the tri-diagonal matrix A
        A = A[:j-1, :j-1]

    # add the final eigenvalue
    evals[m-1] = A

    return evals, total_T_array

def plots_Q3(ApplyShift = False):
    """
    Run the function Submatrix_QR_Alg() on matrices A, B and C and plot the results
    
    :param shift: if True, use the shifted QR algorithm
    """
    # String for plot titles
    if ApplyShift:
        textstr = "    Shifted QR"
    else:
        textstr = "Non-Shifted QR"

    # For each matrix in the dictionary, apply Submatrix_QR_Alg() and plot the results
    for mat in matrices_dict:
        t_array = Submatrix_QR_Alg(matrices_dict[mat].copy(), ApplyShift)[1]
        plt.semilogy(t_array)
        plt.title("Semilog plot of the|T_m,m-1| values ("+ textstr +"), for matrix: " + mat)
        plt.axhline(y=1.0e-12, color='r', linestyle=':')
        plt.xlabel("Iteration Number")
        plt.ylabel("|T_m,m-1|")
        plt.show()

        # Comparison to the pure_QR algorithm from exercises9
        A, its = pure_QR(matrices_dict[mat].copy(), 5000, 1.0e-12, its=True)
        print("Matrix: " + mat)
        print("Total number of iterations until termination ["+ textstr + " Algorithm from Q3] : " + str(len(t_array)))
        print("Total number of iterations until termination [pureQR Algorithm]                 : " + str(its))


"""
This script obtains the results from Q3 of the report
"""
if __name__ == '__main__':
    # Store the matrices used for plotting
    matrices_dict = {"A": getA(), "B": getB(), "C": getC(), "G": getG()}

    """ Apply program to matrix A """
    Q3d()

    """ Generate plots in 3e), 3f) and 3g) """
    plots_Q3()
    plots_Q3(ApplyShift = True)

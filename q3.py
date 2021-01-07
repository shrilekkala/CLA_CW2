import numpy as np
from cla_utils.exercises8 import hessenberg

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

def qr_alg_tri(A, maxit, return_Tarray = False):
    """
    For matrix A, apply the QR algorithm till the stopping criteria and return the result.

    :param A: an mxm symmetric, tridiagonal matrix
    :param maxit: the maximum number of iterations
    :param return_Tarray: logical

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
        # Obtain R and V from the QR factorisation algorithm
        R, V = qr_factor_tri(Ak)
        
        # Find RQ transpose via implicit multplication
        RQ = R.T
        for k in range(0, m):
            if k != m-1:
                RQ[k:k+2,] -= 2 * np.outer(V[:,k], V[:,k]) @ RQ[k:k+2,]
            else:
                RQ[k:k+2,] -= 2 * RQ[k:k+2,]

        # Update variables
        Ak = RQ.T
        its += 1
        Tlist.append(np.abs(Ak[m-1, m-2]))

        # check stopping criteria
        if np.abs(Tlist[-1]) < 1.0e-12:
            print("Stopped after " + str(its) + " iterations")
            break
        elif its+1 > maxit:
            print("Maximum number of iterations reached")
            break
    
    if return_Tarray:
        return Ak, np.array(Tlist)
    else:
        return Ak

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

def Q3d():
    """
    Function to investigate the qr_alg_tri applied to matrix A in Q3 d)
    """
    A1 = getA()
    A2 = A1.copy()
    hessenberg(A2)
    print(qr_alg_tri(A2, 1000))
    print(np.linalg.eig(A1)[0])
    return

# Q3d()






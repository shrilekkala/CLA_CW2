import numpy as np

def triA(c, d, m):
    """
    Creates matrix tri-diagonal matrix A as shown in question 1
    parameters: values c and d and size of matrix (m)

    :return A: the required matrix A
    """
    A = np.eye(m) * c
    D = np.diag(np.ones(m-1) * d, 1)
    A += D + D.T
    return A

def householder_tri(A, kmax=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce
    to upper triangular. If not present, will default to n.

    :return R: an mxn-dimensional numpy array containing the upper
    triangular matrix
    """

    m, n = A.shape

    if kmax is None:
        kmax = n

    # k cycles from 1 to n
    for k in range(1, kmax+1):
        
        x = A[k-1: k+1, k-1]
        
        # initialise e_1 of length m-k+1
        e_1 = np.eye(np.size(x,0))[:,0]

        # householder algorithm
        v_k = np.sign(x[0]) * np.linalg.norm(x) * e_1 + x
        v_k = v_k / np.linalg.norm(v_k)

        A[k-1: k+1, k-1: kmax] = A[k-1: k+1, k-1: kmax]  - 2 * np.outer(v_k, v_k) @ A[k-1: k+1, k-1: kmax]
        
        # for case when kmax =/= n, compute Q*b in place of b
        A[k-1:k+1,kmax:] = A[k-1:k+1,kmax:] - 2 * np.outer(v_k, v_k) @ A[k-1:k+1,kmax:]
    return A


def householder_qr_tri(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the reduced QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxn-dimensional numpy array
    :return R: an nxn-dimensional numpy array
    """
    m, n= A.shape
    
    # create an mxm identity matrix
    I = np.eye(m)

    # construct extended array Ahat
    A_hat = np.hstack((A, I))

    # use housholder to computer Q and R
    A_hat_householder = householder_tri(A_hat, n)

    # extract Q_star and R using slice notation
    R = A_hat_householder[:n, :n]
    Q_star = A_hat_householder[:, n:]

    # construct Q by transposing Q_star and taking the first n columns
    Q = Q_star.T[:, :n]
    return Q, R

A = triA(2,4,6)
A2 = householder_tri(A.copy())
Q2, R2 = householder_qr_tri(A.copy())



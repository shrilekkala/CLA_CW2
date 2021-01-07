import numpy as np

def qr_alg_tri(A):
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
import numpy as np

def triA(c, d, m):
    """
    Creates matrix tri-diagonal matrix A as shown in question 1
    parameters: values c and d and size of matrix (m)

    :return A: the required matrix A
    """
    A = np.eye(m, dtype = 'complex') * c
    D = np.diag(np.ones(m-1, dtype = 'complex') * d, 1)
    A += D + D.T
    return A

"""
Question 1 a)
"""
def LUtri(c, d, m):
    """
    Algorithm 4 from the coursework
    Constructs matrix A and implements LU decomposition on this matrix
    Returns L and U
    """
    U = triA(c, d, m)
    L = np.eye(m, dtype = 'complex')

    for k in range(m-1):
        L[k+1, k] = U[k+1, k] / U[k, k]
        U[k+1, k:k+2] -= L[k+1, k] * U[k, k:k+2]

    return L, U

def forward_sub(L, b):
    """
    Algorithm 5 from the coursework
    Solves Ly=b via forward substitution given L and b
    Returns y
    """
    m, _ = L.shape
    y = np.zeros(m, dtype = 'complex')

    y[0] = b[0] / L[0, 0]
    for k in range(1, m):
        j = k-1
        y[k] = (b[k] - L[k, j] * y[j]) / L[k, k]

    return y

def back_sub(U, y):
    """
    Algorithm 6 from the coursework
    Solves Ux=y via back substitution given U and y
    Returns x
    """
    m, _ = U.shape
    x = np.zeros(m, dtype = 'complex')

    x[m-1] = y[m-1] / U[m-1, m-1]
    for k in range(m-2, -1, -1):
        j = k+1
        x[k] = (y[k] - U[k, j] * x[j]) / U[k, k]

    return x   

"""
Question 1 d)
"""
def LU_solve(c, d, m, b, returnLU = False):
    """
    Algorithm 7 from the coursework
    Given a vector b and a parametrisation of matrix A using c, d and m
    Solves Ax = b for vector x

    :return x: the m dimensional vector solution
    if returnLU = true:
    :return L, U : the L, U matrices of the factorisation of A
    """
    # Initialise matrices L,U and vectors x, y
    U = triA(c, d, m)
    L = np.eye(m, dtype = 'complex')

    y = np.zeros(m, dtype = 'complex')
    x = np.zeros(m, dtype = 'complex')

    # Merged LU factorisation and forward substitution
    y[0] = b[0] / L[0, 0]
    for k in range(m-1):
        L[k+1, k] = U[k+1, k] / U[k, k]
        U[k+1, k:k+2] -= L[k+1, k] * U[k, k:k+2]

        j = k
        y[k+1] = (b[k+1] - L[k+1, j] * y[j]) / L[k+1, k+1]

    # Back Substitution
    x[m-1] = y[m-1] / U[m-1, m-1]
    for k in range(m-2, -1, -1):
        j = k+1
        x[k] = (y[k] - U[k, j] * x[j]) / U[k, k]

    if returnLU:
        return x, L, U
    else:
        return x

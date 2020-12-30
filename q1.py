import numpy as np

"Q1"
def triA(c, d, m):
    A = np.eye(m) * c
    D = np.diag(np.ones(m-1) * d, 1)
    A += D + D.T
    return A

def LUtri(c, d, m):
    A = triA(c, d, m)

    U = A.copy()
    L = np.eye(m)

    for k in range(m-1):
        L[k+1, k] = U[k+1, k] / U[k, k]
        U[k+1, k:k+2] -= L[k+1, k] * U[k, k:k+2]

    return A, L, U

A, L, U = LUtri(3,4,5)


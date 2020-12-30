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

def forward_sub(L, b):
    m, _ = L.shape
    x = np.zeros(m)

    x[0] = b[0] / L[0, 0]
    for k in range(1, m):
        j = k-1
        x[k] = (b[k] - L[k, j] * x[j]) / L[k, k]

    return x

def back_sub(U, y):
    m, _ = U.shape
    x = np.zeros(m)

    x[m-1] = y[m-1] / U[m-1, m-1]
    for k in range(m-2, -1, -1):
        j = k+1
        x[k] = (y[k] - U[k, j] * x[j]) / U[k, k]

    return x   

A, L, U = LUtri(3,4,5)
b = np.random.rand(5)
x = forward_sub(L, b)
x2 = back_sub(U, b)
U @ x2
b

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

c = 5
d = 6
m = 10
A, L, U = LUtri(c,d,m)
b = np.random.rand(m)
x1 = forward_sub(L, b)

x = back_sub(U, x1)
A@x
b

def LU_solve(c, d, m, b):
    A = triA(c, d, m)

    U = A.copy()
    L = np.eye(m)

    y = np.zeros(m)
    y[0] = b[0] / L[0, 0]

    for k in range(m-1):
        L[k+1, k] = U[k+1, k] / U[k, k]
        U[k+1, k:k+2] -= L[k+1, k] * U[k, k:k+2]

        j = k
        y[k+1] = (b[k+1] - L[k+1, j] * y[j]) / L[k+1, k+1]

    x = np.zeros(m)
    x[m-1] = y[m-1] / U[m-1, m-1]

    for k in range(m-2, -1, -1):
        j = k+1
        x[k] = (y[k] - U[k, j] * x[j]) / U[k, k]

    return A, L, U, x

c = 2
d = 6
m = 7
b = np.random.rand(m)
A, L, U, x = LU_solve(c, d, m, b)
A@x
b


def LU_inplace(c, d, m):
    A1 = triA(c, d, m)

    A = A1.copy()

    for k in range(m-1):
        A[k+1, k] = A[k+1, k] / A[k, k]
        A[k+1, k+1] = A[k+1, k+1] - np.outer(A[k+1, k], A[k, k+1])
    
    return(A)
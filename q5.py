import numpy as np
import matplotlib.pyplot as plt

"""
Question 5 f)
"""

# construct matrix V
def getV(N, alpha):
    x = np.arange(N)
    y = alpha ** (-x/N)
    z = np.zeros((N, N), dtype = 'complex')
    for k in range(N):
        z[:, k] = np.exp(1j * 2 * np.pi * x * k / N)
        z[:, k] = np.multiply(y, z[:, k])
    
    return z

def same_block_diag(A, k):
    """
    Function that creates a block diagonal matrix (which has k blocks each being A)
    """
    m, _ = A.shape
    
    # construct the block diagonal matrix
    B = np.zeros((k*m,k*m))
    for i in range(k):
        B[i*m:(i+1)*m, i*m:(i+1)*m] = A

    return B


# construct the diagonal matrix used in 5d)
def getD(M, N, alpha):
    x = np.arange(N)
    diag = N * (alpha ** (-x/N))
    D = np.eye(2*M*N)

    for i in range(N):
        D[i * 2*M : (i+1) * 2*M, i * 2*M : (i+1) * 2*M] *= diag[i]

    return D


# algorithm for solving [V(x)I]U
def step3(M, N, U, alpha):
    D = getD(M, N, alpha)

    Uprime = U.reshape(N, 2*M).T
    Uifft = np.fft.ifft(Uprime).reshape(2*M*N, order='F')
    
    VIU = D @ Uifft

    return VIU

M = 2
N = 3
alpha = 0.1
U = np.arange(2*M*N)+1
Uprime = U.reshape(N, 2*M).T
Uifft = np.fft.ifft(Uprime).reshape(2*M*N, order='F')
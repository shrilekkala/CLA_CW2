import numpy as np
import matplotlib.pyplot as plt

"""
Question 5 f)
"""
def getD(M, N, alpha, inverse = False):
    """
    Function that constructs the block diagonal matrix D or D inverse from 5d)
    """
    # construct the coefficients of the blocks
    x = np.arange(N)
    diag = N * (alpha ** (-x/N))
    D = np.eye(2*M*N)

    # construct the matrix D or D inverse as required
    for i in range(N):
        if inverse:
            D[i * 2*M : (i+1) * 2*M, i * 2*M : (i+1) * 2*M] /= diag[i]
        else:
            D[i * 2*M : (i+1) * 2*M, i * 2*M : (i+1) * 2*M] *= diag[i]

    return D


def step3(M, N, U, alpha):
    """
    Algorithm for step 3 Q5e)
    Constructs and returns [V(x)I]U as in 5 d)
    """
    # get block-diagonal matrix D
    D = getD(M, N, alpha)

    # reshape U
    Uprime = U.reshape(N, 2*M).T
    
    # Apply the IFFT algorithm and reshape back
    Uifft = np.fft.ifft(Uprime).reshape(2*M*N, order='F')
    
    # obtain [V(x)I]U by matrix multiplication
    VIU = D @ Uifft

    return VIU

def step1(M, N, R, alpha):
    """
    Algorithm for step 1 Q5e)
    Constructs and returns [V^{-1}(x)I]R as in 5 d)
    """    
    # get block-diagonal matrix D inverse
    Dinv = getD(M, N, alpha, inverse = True)

    # obtain D^{-1} R by matrix multiplication
    DinvR = Dinv @ R

    # reshape as required
    DinvR = DinvR.reshape((N, 2*M)).T

    # Apply the IFFT algorithm and reshape back
    VinvIR = np.fft.fft(DinvR).reshape(2*M*N, order='F')
    
    return VinvIR


"""
M = 2
N = 3
alpha = 0.1
U = np.arange(2*M*N)+1
"""
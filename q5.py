import numpy as np
import matplotlib.pyplot as plt
from q1 import solver_alg

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

def step2(d1, d2, delt, delx, rk):
    M = int(rk.shape[0] / 2)

    rk1 = rk[:M]
    rk2 = rk[M:]

    c1 = (d2*delt)**2 / (delx * d1**2)
    b = d2*B_21@rk1 - d1*rk2

    # get qk from almost tridiagonal system solver from q2
    qk = solver_alg(c1, b/(-d1**2))

    # get pk by substitution
    pk = (rk1 - d2 * (- delt) * qk) / d1

    pq = np.concatenate((pk, qk))

    return pq

def getB_12(delt, M):
    B_12 = np.diag(-np.ones(M)* delt)
    return B_12

def getB_21(delt, delx, M):
    A = np.diag(np.ones(M-1)*(- delt / delx), 1)
    A[0, M-1] = - delt / delx
    D = np.diag(np.ones(M)*(2*delt/delx))
    B_21 = D + A + A.T
    return B_21

def getB(B_12, B_21):
    M, _ = B_12.shape
    B = np.zeros((2*M,2*M))
    B[M:,:M] = B_21
    B[:M,M:] = B_12
    return B

def getT(d1, d2, delt, delx, M):
    a = -(2*(d2*delt)**2)/delx - d1**2
    b = ((d2*delt)**2)/delx
    A = np.diag(np.ones(M-1)*b, 1)
    A[0, M-1] = b
    D = np.diag(np.ones(M)*a)
    T = D + A + A.T
    return T

M = 5
delt = 5
delx = 10
B_12 = getB_12(delt, M)
B_21 = getB_21(delt, delx, M)
B = getB(B_12, B_21)
I = np.eye(2*M)
d1 = 2
d2 = 3
Mat = d1*I + d2*B

rk = np.random.randn(2*M)
pq = np.linalg.solve(Mat, rk)

rk1 = rk[:M]
rk2 = rk[M:]
Ta = (d2**2) * (-delt) * B_21 - (d1**2) * np.eye(M)
T = getT(d1, d2, delt, delx, M)

b = d2*B_21@rk1 - d1*rk2
qk1 = np.linalg.solve(T, b)

c1 = (d2*delt)**2 / (delx * d1**2)

# check
A = q1.triA(1+2*c1, -c1, M)
A[0, M-1] = -c1
A[M-1, 0] = -c1
T / (-d1**2) - A


pq1 = step2(d1, d2, delt, delx, rk)






"""
M = 2
N = 3
alpha = 0.1
U = np.arange(2*M*N)+1
"""


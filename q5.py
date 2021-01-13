import numpy as np
import matplotlib.pyplot as plt
from q2 import solver_alg

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

def getB_21(delt, delx, M):
    """
    Function that constructs the B_21 sub-block of matrix B from 5a)
    """
    # construct the sub-diagonals and corners
    A = np.diag(np.ones(M-1)*(- delt / delx), 1)
    A[0, M-1] = - delt / delx

    # construct the main diagonal
    D = np.diag(np.ones(M)*(2*delt/delx))

    B_21 = D + A + A.T
    return B_21

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
    """
    Algorithm for step 2 Q5e)
    Constructs and returns (pk^T, qk^T), i.e. the kth time slice of U hat
    """    
    M = int(rk.shape[0] / 2)

    # get sub-block B_21 of matrix B
    B_21 = getB_21(delt, delx, M)

    # split rk into 2 equal length vectors
    rk1 = rk[:M]
    rk2 = rk[M:]

    # get constant c1 and vector b as required
    c1 = (d2*delt)**2 / (delx * d1**2)
    b = d2*B_21@rk1 - d1*rk2

    # Find qk from almost tridiagonal system solver from q2
    qk = solver_alg(c1, b/(-d1**2))

    # Find pk by substitution
    pk = (rk1 - d2 * (- delt) * qk) / d1

    # Join pk and qk into one vector
    pq = np.concatenate((pk, qk))

    return pq
 
"""
---------------
"""
def getB(delt, delx, M):
    """
    Function that constructs the matrix B of eigenvectors as in 5 a)
    """
    # construct the non-zero blocks of B
    B_12 = np.diag(-np.ones(M)* delt)
    B_21 = getB_21(delt, delx, M)

    # combine the blocks as required
    B = np.zeros((2*M,2*M))
    B[M:,:M] = B_21
    B[:M,M:] = B_12
    return B

def getC1a(M, alpha):
    D1 = np.diag(np.ones(M))
    D2 = np.diag(-np.ones(M-1), -1)
    D1[0, M-1] = -alpha
    C1a = D1 + D2
    return C1a

M = 4
N = 3
alpha = 0.1
delx = 0.5
delt = 0.1


U = np.arange(2*M*N)+1

# Check step 17 works

# LHS matrix
C1a = getC1a(N, alpha)
C2a = C1a.copy()/2
I = np.eye(2*M)
B = getB(delt, delx, M)
Mat = np.kron(C1a, I) + np.kron(C2a, B)

# RHS vector
r = np.random.randn(2*M)
pq = U[-2*M:]
topR = r + alpha * (-I + B/2) @ pq
R = np.zeros(2*M*N)
R[:2*M] = topR

# get x via numpy functions
x = np.linalg.solve(Mat, R)

# Step 1
Rhat = step1(M, N, R, alpha)

# Step 2
x_range = np.arange(N)
diag = N * (alpha ** (-x_range/N))

lambdas = 1 - alpha**(1/N) * np.exp((2*(N-1)/N)*np.pi*1j*x_range)
D1 = np.diag(lambdas)
D2 = np.diag(lambdas/2)

Uhat = np.zeros(2*M*N, dtype = 'complex')

for k in range(N):
    print(k)
    rk = Rhat[k * (2*M) : (k + 1) * (2*M)]
    print(rk)

    d1 = D1[k,k]
    d2 = D2[k,k]

    newpq = step2(d1, d2, delt, delx, rk)
    Uhat[k * (2*M) : (k + 1) * (2*M)] = newpq

LM = np.kron(D1, I) + np.kron(D2, B)
Rv = Rhat.copy()
x1 = np.linalg.solve(LM, Rv)

# Step 3
Uk1 = step3(M, N, x1, alpha)

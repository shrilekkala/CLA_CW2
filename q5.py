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
    Algorithm for step 2 Q5e) for one time slice
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
 
def eq17(M, N, delx, delt, Uk, alpha, r):
    """
    Algorithm to compute U^(k+1) from U^k as in equation 17
    
    :param delx: space steps
    :param delt: time steps
    :param M: number of space steps required
    :param N: number of time steps required
    :param Uk: as in the question  
    
    :param alpha: constant in corner of C_1^(alpha) and C_2^(alpha)
    :param r: vector r from Q5a)

    :return Uk1: the solution to (17)
    """
    # Construct B and I
    pq = Uk[-2*M:]
    I = np.eye(2*M)
    B = getB(delt, delx, M)
    topR = r + alpha * (-I + B/2) @ pq

    # Construct R
    R = np.zeros(2*M*N, dtype = 'complex')
    R[:2*M] = topR

    # Apply Step 1
    Rhat = step1(M, N, R, alpha)

    # Apply Step 2
    x_range = np.arange(N)
    lambdas = 1 - alpha**(1/N) * np.exp((2*(N-1)/N)*np.pi*1j*x_range)
    mu_s = (1 + alpha**(1/N) * np.exp((2*(N-1)/N)*np.pi*1j*x_range)) / 2
    D1 = np.diag(lambdas)
    D2 = np.diag(mu_s)

    Uhat = np.zeros(2*M*N, dtype = 'complex')

    # Loop through the time slices
    for k in range(N):
        rk = Rhat[k * (2*M) : (k + 1) * (2*M)]

        d1 = D1[k,k]
        d2 = D2[k,k]

        # Apply the step2 function
        Uhat[k * (2*M) : (k + 1) * (2*M)] = step2(d1, d2, delt, delx, rk)

    # Apply Step 3
    Uk1 = step3(M, N, Uhat, alpha)

    return Uk1

def U_solver_alg(U0, M, N, delx, delt, alpha, r):
    """
    Algorithm to compute U iteratively from the initial guess as in equation 17
    
    :param U0: initial guess for the solution
    :param delx: space steps
    :param delt: time steps
    :param M: number of space steps required
    :param N: number of time steps required    
    :param alpha: constant in corner of C_1^(alpha) and C_2^(alpha)
    :param r: vector r from Q5a)

    :return Uk1: the solution to (17)
    """
    maxits = 1000
    counter = 0
    Uk = U0

    while True:
        counter += 1
        Uk1 = eq17(M, N, delx, delt, Uk, alpha, r)

        # Check for convergence
        if np.allclose(Uk, Uk1, rtol = 1e-12) == True:
            break
        elif counter > maxits:
            break

        Uk = Uk1
    
    return Uk1, counter

"""
M = 4
N = 3
alpha = 0.01
delx = 0.1
delt = 0.1
r = np.random.randn(2*M)

U0 = np.zeros(2*M*N)
Un, k = U_solver_alg(U0, M, N, delx, delt, alpha, r)
"""
def w_func(x,t):
    w_vals = (np.sin(2 * np.pi * x) / (2 * np.pi)) * np.cos(2 * np.pi * t)
    return w_vals

def u_func(x,t):
    u_vals = (np.sin(2 * np.pi * x) / (2 * np.pi)) * (np.sin(2 * np.pi * t) / (2 * np.pi))
    return u_vals

M = 10
delx = 1/M
delt = 1/1000
N = 251
alpha = 0.1

# Initial Values at time 0
x_vals = (np.arange(M)+1) * delx
u0 = u_func(x_vals,0)
w0 = w_func(x_vals,0)


# Create array r
r = np.zeros(2*M)
r[:M] = u0 + delt * w0 / 2
r[M] = w0[0] + (delt / (2*delx)) * (u0[-1] - 2*u0[0] + u0[1])
r[M+1:2*M-1] = w0[1:M-1] + (delt / (2*delx)) * (u0[0:M-2] - 2*u0[1:M-1] + u0[2:M])
r[2*M - 1] = w0[M-1] + (delt / (2*delx)) * (u0[-2] - 2*u0[-1] + u0[0])

# Initial Guess
U0 = np.zeros(2*M*N)
U, k = U_solver_alg(U0, M, N, delx, delt, alpha, r)

# Change U into a 2M x N matrix
U = U.reshape(N, 2*M).T
U = np.real(U)

x_range = np.linspace(x_vals[0],1,501)

for i in range(5):
    k = 50*i
    pk = U[:M, k]
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    plt.plot(x_vals, pk, color = color)
    actual_u_vals = u_func(x_range, k * delt) 
    plt.plot(x_range, actual_u_vals, ls=':', color = "black")
    # plt.show()
plt.show()

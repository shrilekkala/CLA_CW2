''' test script for q3 '''
import numpy as np
import pytest
import q5

# construct matrix V
def getV(N, alpha):
    """
    Function that constructs the matrix V of eigenvectors as in 5 c)
    """
    x = np.arange(N)
    V = np.zeros((N, N), dtype = 'complex')

    # obtain the coefficients of the blocks
    y = alpha ** (-x/N)

    # construct V column by column
    for k in range(N):
        V[:, k] = np.exp(1j * 2 * np.pi * x * k / N)
        V[:, k] = np.multiply(y, V[:, k])
    
    return V

# construct matrix B
def getB(delt, delx, M):
    """
    Function that constructs the matrix B of eigenvectors as in 5 a)
    """
    # construct the non-zero blocks of B
    B_12 = np.diag(-np.ones(M)* delt)
    B_21 = q5.getB_21(delt, delx, M)

    # combine the blocks as required
    B = np.zeros((2*M,2*M))
    B[M:,:M] = B_21
    B[:M,M:] = B_12
    return B


''' 
Test the step3 function
'''
@pytest.mark.parametrize('M, N, alpha', [(2, 3, 0.1), (5, 5, 0.2), (8, 10, 0.15)])
def test_step3(M, N, alpha):
    np.random.seed(1234*M)

    # Create random U
    U = np.random.randn(2*M*N)

    # First obtain (V(x)I) U via the step3 function
    VIU = q5.step3(M, N, U, alpha)

    # Obtain (V(x)I) U directly by constructing V and I and using np.kron
    V = getV(N, alpha)
    I = np.eye(2*M)
    VIU2 = np.kron(V, I) @ U

    # Check that the error is within a threshold
    err = VIU - VIU2
    assert(np.linalg.norm(err) < 1.0e-6)

''' 
Test the step1 function
'''
@pytest.mark.parametrize('M, N, alpha', [(2, 3, 0.1), (5, 5, 0.2), (8, 10, 0.15)])
def test_step1(M, N, alpha):
    np.random.seed(5678*M)

    # Create random U
    R = np.random.randn(2*M*N)

    # First obtain (V^{-1}(x)I) U via the step1 function
    VIinvR = q5.step1(M, N, R, alpha)

    # Obtain (V^{-1}(x)I) U directly by constructing V and I and using np.kron
    V = getV(N, alpha)
    I = np.eye(2*M)
    VIinvR2 = np.kron(np.linalg.inv(V), I) @ R

    # Check that the error is within a threshold
    err = VIinvR - VIinvR2
    assert(np.linalg.norm(err) < 1.0e-6)

''' 
Test the step2 function
'''
@pytest.mark.parametrize('M, delt, delx', [(5, 5, 10), (10, 2, 3), (20, 0.5, 0.1)])
def test_step2(M, delt, delx):
    np.random.seed(2468*M)

    d1 = np.random.randint(1,10)
    d2 = np.random.randint(1,10)
    rk = np.random.randn(2*M)

    # Construct B
    B = getB(delt, delx, M)
    I = np.eye(2*M)
    Mat = d1*I + d2*B

    pq1 = np.linalg.solve(Mat, rk)

    pq2 = q5.step2(d1, d2, delt, delx, rk)

    # Check that the error is within a threshold
    err = pq1 - pq2
    assert(np.linalg.norm(err) < 1.0e-6)


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

# construct matrix C_1^(alpha)
def getC1a(M, alpha):
    """
    Function that constructs the matrix C_1^(alpha) as in 5 b)
    """
    D1 = np.diag(np.ones(M))
    D2 = np.diag(-np.ones(M-1), -1)
    D1[0, M-1] = -alpha
    C1a = D1 + D2
    return C1a

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

    # Create random d1, d2 and vector rk
    d1 = np.random.randint(1,10) + 1j
    d2 = np.random.randint(1,10) + 2j
    rk = np.random.randn(2*M)

    # Get B and I and form the matrix d1*I + d2*B
    B = q5.getB(delt, delx, M)
    I = np.eye(2*M)
    Mat = d1*I + d2*B

    # Solve for the solution using numpy functions
    pq1 = np.linalg.solve(Mat, rk)

    # Solve for the solution using the step2 function
    pq2 = q5.step2(d1, d2, delt, delx, rk)

    # Check that the error is within a threshold
    err = pq1 - pq2
    assert(np.linalg.norm(err) < 1.0e-6)

''' 
Test the eq17 function
'''
@pytest.mark.parametrize('M, N, alpha', [(4, 3, 0.2), (10, 5, 0.5), (20, 15, 0.25)])
def test_eq17(M, N, alpha):
    np.random.seed(1066*M)

    # create random grid spacing for timesteps and spacesteps
    delx = np.random.randint(1,10)/10
    delt = np.random.randint(1,10)/10

    # Create random vectors U^k and r
    Uk = np.arange(2*M*N)+1
    r = np.random.randn(2*M)

    # Obtain U^(k+1) from the eq17 function
    x1 = q5.eq17(M, N, delx, delt, Uk, alpha, r)

    # Construct the LHS matrix from (17)
    C1a = getC1a(N, alpha)
    C2a = C1a.copy()/2
    I = np.eye(2*M)
    B = q5.getB(delt, delx, M)
    Mat = np.kron(C1a, I) + np.kron(C2a, B)

    # Construct the RHS vector from (17)
    pq = Uk[-2*M:]
    topR = r + alpha * (-I + B/2) @ pq
    R = np.zeros(2*M*N)
    R[:2*M] = topR

    # Obtain U^(k+1) using numpy functions
    x2 = np.linalg.solve(Mat, R)

    # Check that the error is within a threshold
    err = x1 - x2
    assert(np.linalg.norm(err) < 1.0e-6)
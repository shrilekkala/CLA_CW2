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
    y = alpha ** (-x/N)
    z = np.zeros((N, N), dtype = 'complex')
    for k in range(N):
        z[:, k] = np.exp(1j * 2 * np.pi * x * k / N)
        z[:, k] = np.multiply(y, z[:, k])
    
    return z

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
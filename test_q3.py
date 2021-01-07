''' test script for q3 '''
import numpy as np
import pytest
import q3

''' 
Test the qr_factor_tri function from Q2 part c)
'''
@pytest.mark.parametrize('m', [14, 20, 135, 500, 935])
def test_qr_factor_tri(m):
    np.random.seed(5501*m)

    # Create a random, symmetric tridiagonal matrix A
    d1 = np.random.rand(m)
    d2 = np.random.rand(m-1)
    A = np.diag(d1) + np.diag(d2,1) + np.diag(d2,-1)

    # Obtain R and V from qr_alg_tri
    R, V = q3.qr_factor_tri(A.copy())

    # Construct Q from V
    Q = np.eye(m)
    for k in range(0, m):
        if k != m-1:
            Q[k:k+2, :] = Q[k:k+2, :] - 2 * np.outer(V[:,k], V[:,k]) @ Q[k:k+2, :]
        else:
            Q[k:k+2, :] = Q[k:k+2, :] - 2 *  Q[k:k+2, :]
    Q = Q.T

    # Find the error in the factorisation
    err = A - Q@R

    # Check that the the correct Q and R are obtained
    assert(np.linalg.norm(err) < 1.0e-6)

''' 
Test the qr_alg_tri function from Q2 part d)
'''
@pytest.mark.parametrize('m', [14, 20, 135, 500, 935])
def test_qr_alg_tri(m):
    np.random.seed(6601*m)

    # Create a random, symmetric tridiagonal matrix A
    d1 = np.random.rand(m)
    d2 = np.random.rand(m-1)
    A = np.diag(d1) + np.diag(d2,1) + np.diag(d2,-1)

    # Obtain Ak from qr_alg_tri
    Ak = q3.qr_alg_tri(A.copy(), 1000)

    # Check it is still Hermitian
    assert(np.linalg.norm(Ak - np.conj(Ak).T) < 1.0e-4)

    # Check that the m, m-1 element is of appropriate size
    assert(Ak[m-1,m-2] < 1.0e-12)

    # Check its still tridiagonal
    assert(np.linalg.norm(Ak[np.tril_indices(m, -2)])/m**2 < 1.0e-5)

    # Check for conservation of trace
    assert(np.abs(np.trace(A) - np.trace(Ak)) < 1.0e-6)
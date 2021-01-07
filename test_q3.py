''' test script for q3 '''
import numpy as np
import pytest
import q3

''' 
Test the qr_factor_tri function from Q2 part c)
'''
@pytest.mark.parametrize('m', [14, 20, 135, 500, 935])
def test_solver_alg(m):
    np.random.seed(5501*m)

    # Create a random, symmetric tridiagonal matrix A
    d1 = np.random.rand(m)
    d2 = np.random.rand(m-1)
    A = np.diag(d1) + np.diag(d2,1) + np.diag(d2,-1)

    # Obtain R and V from qr_alg_tri
    R, V = q3.qr_alg_tri(A.copy())

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
''' test script for q4 '''
import numpy as np
import pytest
import q4
from cla_utils.exercises5 import solve_R

''' 
Test the GMRES function from Q4 part a) 
Using no preconitioning (so b tilde = b)
'''
@pytest.mark.parametrize('m', [15, 35, 61, 101, 204])
def test_GMRES1(m):
    np.random.seed(5501*m)
    A = np.random.randn(m, m)
    b = np.random.randn(m)

    # Here M is the identity
    def apply_pc1(x):
        return x

    x, _ = q4.GMRES(A, b, apply_pc1, maxit=1000, tol=1.0e-3)
    assert(np.linalg.norm(np.dot(A, x) - b) < 1.0e-3)

''' 
Test the GMRES function from Q4 part a) 
Using preconditioning matrix M = diag(A)
'''
@pytest.mark.parametrize('m', [15, 35, 61, 101, 204])
def test_GMRES2(m):
    np.random.seed(5501*m)
    A = np.random.randn(m, m)
    b = np.random.randn(m)

    # Here M is the is just diag(A)
    def apply_pc2(y):
        y_tilde = y / np.diag(A)
        return y_tilde

    x, _ = q4.GMRES(A, b, apply_pc2, maxit=1000, tol=1.0e-3)
    assert(np.linalg.norm(np.dot(A, x) - b) < 1.0e-3)

''' 
Test the GMRES function from Q4 part a) 
Using preconditioning matrix M = upper triangular part of A
'''
@pytest.mark.parametrize('m', [5, 6, 8, 9, 15])
def test_GMRES3(m):
    np.random.seed(1024*m)
    A = np.random.randn(m, m)
    b = np.random.randn(m)

    # Here M is the upper triangular part of A
    def apply_pc3(y):
        M = np.triu(A)
        y_tilde = solve_R(M, y)
        return y_tilde

    x, _ = q4.GMRES(A, b, apply_pc3, maxit=1000, tol=1.0e-3)
    assert(np.linalg.norm(np.dot(A, x) - b) < 1.0e-3)


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
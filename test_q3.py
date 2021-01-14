''' test script for q3 '''
import numpy as np
import pytest
import q3

''' 
Test the qr_factor_tri function from Q3 part c)
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

    # Check that the correct Q and R are obtained
    assert(np.linalg.norm(err) < 1.0e-6)

''' 
Test the qr_alg_tri function from Q3 part d)
'''
@pytest.mark.parametrize('m', [14, 20, 55, 126, 201])
def test_qr_alg_tri(m):
    np.random.seed(6601*m)

    # Create a random, symmetric tridiagonal matrix A
    d1 = np.random.rand(m)
    d2 = np.random.rand(m-1)
    A = np.diag(d1) + np.diag(d2,1) + np.diag(d2,-1)

    # Obtain Ak from qr_alg_tri
    Ak = q3.qr_alg_tri(A.copy(), 1000)

    # Check it is still symmetric
    assert(np.linalg.norm(Ak - Ak.T) < 1.0e-4)

    # Check its still tridiagonal
    assert(np.linalg.norm(Ak[np.tril_indices(m, -2)])/m**2 < 1.0e-5)

    # Check for conservation of trace
    assert(np.abs(np.trace(A) - np.trace(Ak)) < 1.0e-6)

    # Check that the m, m-1 element is of appropriate size
    assert(Ak[m-1,m-2] < 1.0e-12)

''' 
Test the Submatrix_QR_Alg() function from Q3 part e)
'''
@pytest.mark.parametrize('m', [10, 15, 21, 26, 30])
def test_Submatrix_QR_Alg(m):
    np.random.seed(4567*m)

    # Create a random, symmetric matrix A
    A = np.random.rand(m, m)
    A = 1/2 * (A + A.T)

    # Obtain the eigenvalues of A
    evals1 = np.linalg.eigvals(A)

    # Obtain the eigenvalues of A from the function from Q3e()
    evals2 = q3.Submatrix_QR_Alg(A)[0]

    # Sort the arrays into increasing order
    evals1.sort()
    evals2.sort()

    # Check that the eigenvalues are the same to a certain threshold
    assert(np.allclose(evals1, evals2, 1.0e-6))

''' 
Test the shifted qr_alg_tri function from Q3 part f)
'''
@pytest.mark.parametrize('m', [14, 20, 55, 126, 201])
def test_shifted_qr_alg_tri(m):
    np.random.seed(6601*m)

    # Create a random, symmetric tridiagonal matrix A
    d1 = np.random.rand(m)
    d2 = np.random.rand(m-1)
    A = np.diag(d1) + np.diag(d2,1) + np.diag(d2,-1)

    # Obtain Ak from the shifted version of qr_alg_tri
    Ak = q3.qr_alg_tri(A.copy(), 1000, shift = True)

    # Check it is still symmetric
    assert(np.linalg.norm(Ak - Ak.T) < 1.0e-4)

    # Check that the m, m-1 element is of appropriate size
    assert(Ak[m-1,m-2] < 1.0e-12)

    # Check its still tridiagonal
    assert(np.linalg.norm(Ak[np.tril_indices(m, -2)])/m**2 < 1.0e-5)

    # Check for conservation of trace
    assert(np.abs(np.trace(A) - np.trace(Ak)) < 1.0e-6)


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
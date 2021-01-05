''' test script for q1 '''
import pytest
import cla_utils
import q1
import numpy as np

''' 
Test the 3 algorithms from Q1 part a)
'''
@pytest.mark.parametrize('m', [5, 11, 24, 102, 606])
def test_LUsolve_a(m):
    np.random.seed(5601*m)
    
    # Generate distinct, non-zero random c and d
    c = np.random.randint(1, m**2)
    d = np.random.randint(1, m**2)
    while d == c:
        d = np.random.randint(m)

    # create matrix A using the paramteters
    A = q1.triA(c, d, m)

    # obtain matrices L and U
    L, U = q1.LUtri(c,d,m)

    # Check that L is lower triangular to a specific threshold
    assert(np.allclose(np.tril(L), L, 1.0e-6))

    # Check that U is upper triangular to a specific threshold
    assert(np.allclose(np.triu(U), U, 1.0e-6))

    b = np.random.rand(m)
    y = q1.forward_sub(L, b)
    x = q1.back_sub(U, y)

    err = A@x - b
    # Check that the the correct x is obtained
    assert(np.linalg.norm(err) < 1.0e-6)

''' 
Test the LU solve algorithm from Q1 part d)
'''
@pytest.mark.parametrize('m', [12, 25, 231, 500, 845])
def test_LUsolve_d(m):
    np.random.seed(8564*m)

    # Generate distinct, non-zero random c and d
    c = np.random.randint(1, m**2)
    d = np.random.randint(1, m**2)
    while d == c:
        d = np.random.randint(m)
    
    # create matrix A using the paramteters
    A = q1.triA(c, d, m)

    # create a random vector b
    b = np.random.rand(m)
    
    # obtain x via the function LU_solve
    x = q1.LU_solve(c, d, m, b)

    err = A@x - b
    # Check that the the correct x is obtained
    assert(np.linalg.norm(err) < 1.0e-6)


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
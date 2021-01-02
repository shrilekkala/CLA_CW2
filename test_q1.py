''' test script for q1 '''
import pytest
import cla_utils
import q1
import numpy as np

''' 
Test the algorithms from Q1 part a)
'''
@pytest.mark.parametrize('c, d, m', [(5, 8, 6), (52, 102, 61), (1, 812, 150)])
def test_LUsolve_a(c, d, m):
    np.random.seed(8564*m)
    A = q1.triA(c, d, m)
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


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
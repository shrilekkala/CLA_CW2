''' test script for q2 '''
import pytest
import cla_utils
import q1
import q2
import numpy as np

''' 
Test the solve_algorithm function from Q2 part e)
'''
@pytest.mark.parametrize('m', [14, 20, 135, 500, 935])
def test_solver_alg(m):
    np.random.seed(4501*m)

    # Generate non-zero random c1
    c1 = np.random.randint(1, m**2)

    # construct matrix A
    A = q1.triA(1+2*c1, -c1, m)
    A[0, m-1] = -c1
    A[m-1, 0] = -c1

    # create a random vector b
    b = np.random.rand(m)
    
    # obtain x via the function solver_algorithm
    x = q2.solver_alg(c1, b)

    err = A@x - b

    # Check that the the correct x is obtained
    assert(np.linalg.norm(err) < 1.0e-6)
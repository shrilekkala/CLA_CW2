import numpy as np
import time
import q1
from cla_utils import exercises7

def getBC(c1, L, U):
    """
    Given parameter c1 and matrices L, U as in Q2 d),
    Obtain vectors u3, u4, v3, v4.
    Construct and return matrices B and C
    """
    m, _ = L.shape
    Im = np.eye(m)
    
    # Construct vectors u1 and v1
    u1 = -c1 * Im[:, 0]
    v1 = Im[:,-1]

    # Obtain u3 and u4 via forward substitution with L
    u3 = q1.forward_sub(L, u1)
    u4 = q1.forward_sub(L, v1)

    # Obtain v3 and v4 via forward substitution with U^T
    v3 = q1.forward_sub(U.T, v1)
    v4 = q1.forward_sub(U.T, u1)

    # Construct matrices B and C
    B = np.hstack((u3.reshape(m,1), u4.reshape(m,1)))
    C = np.vstack((v3, v4))

    return B, C


def solver_alg(c1, b):
    """
    Algorithm to solve Ax = b in Q2 f)

    :param c1: the constant c1 in matrix m x m matrix A (and T)
    :param b: the known m dimensional vector in Ax = b

    :return x: an m dimensional vector solution
    """
    m = b.size
    I2 = np.eye(2)

    # Solve Ty = b via LU the factorisation algorithm and obtain LU
    y, L, U = q1.LU_solve(1+2*c1, -c1, m, b, True)

    # Construct the required matrices B and C
    B, C = getBC(c1, L, U)

    # Solve Lr = b via forward substitution
    r = q1.forward_sub(L, b)

    # Construct vector q
    q = B @ np.linalg.inv(I2 + C @ B) @ C @ r

    # Solve Up = q via back substitution
    p = q1.back_sub(U, q)

    # Construct the final x
    x  = y - p

    return x

def compare_algs(m):
    c1 = 20

    # construct matrix A
    A = q1.triA(1+2*c1, -c1, m)
    A[0, m-1] = -c1
    A[m-1, 0] = -c1

    # create a random vector b
    b = np.random.rand(m)

    start_t = time.time()
    # obtain x1 via the above solver_alg function
    x1 = solver_alg(c1, b)
    print("Time taken for solver_alg : " + str(time.time() - start_t))

    start_t = time.time()
    # obtain x2 via the solve_LUP function from exercises 7
    x2 = exercises7.solve_LUP(A.copy(), b)
    print("Time taken for solve_LUP  : " + str(time.time() - start_t))

    return

import numpy as np
import q1

def getBC(u1, v1):
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


def solve_algorithm(c1, b):
    """
    Algorithm to solve Ax = b in Q2 f)

    :param c1: the constant c1 in matrix m x m matrix A (and T)
    :param b: the known m dimensional vector in Ax = b

    :return x: an m dimensional vector solution
    """
    m = b.size
    Im = np.eye(m)
    I2 = np.eye(2)

    # Construct vectors u1 and v1
    u1 = -c1 * Im[:, 0]
    v1 = Im[:,-1]

    # Construct the required matrices B and C
    B, C = getBC(u1, v1)

    # Solve Ty = b via LU the factorisation algorithm and obtain LU
    y, L, U = q1.LU_solve(1+2*c1, -c1, m, b, True)

    # Solve Lr = b via forward substitution
    r = q1.forward_sub(L, b)

    # Construct vector q
    q = B @ np.linalg.inv(I2 + C @ B) @ C @ r

    # Solve Up = q via back substitution
    p = q1.back_sub(U, q)

    # Construct the final x
    x  = y - p

    return x
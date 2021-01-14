import numpy as np
import time
import matplotlib.pyplot as plt
import q1
from cla_utils.exercises6 import LU_inplace
from cla_utils.exercises7 import solve_LUP

"""
Question 2 c)
"""
def Q2c(c1, m):
    """
    Given parameters c1 and m, construct A
    Run the LU factorisation algorithm from exercises 6
    To observe what is happening at each step as required in Q2c
    """
    # construct A
    A = q1.triA(1+2*c1, -c1, m)
    A[0, m-1] = -c1
    A[m-1, 0] = -c1

    # Apply the LU factorisation algorithm
    LU_inplace(A, printsteps = True)

    return

"""
Question 2 f)
"""
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
    I2 = np.eye(2, dtype = 'complex')

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
    """
    Function that compares the times taken for the solver_alg and solve_LUP functions for a given m
    """
    c1 = 20

    # construct matrix A
    A = q1.triA(1+2*c1, -c1, m)
    A[0, m-1] = -c1
    A[m-1, 0] = -c1
    A = np.real(A)

    # create a random vector b
    b = np.random.rand(m)

    start_t = time.time()
    # obtain x1 via the above solver_alg function
    x1 = solver_alg(c1, b)
    print("Time taken for solver_alg (m = " + str(m) +  "): " + str(time.time() - start_t))

    start_t = time.time()
    # obtain x2 via the solve_LUP function from exercises 7
    x2 = solve_LUP(A.copy(), b)
    print("Time taken for solve_LUP  (m = " + str(m) +  "): " + str(time.time() - start_t))

    return

"""
Question 2 g)
"""
def get_f(u, w, delx, delt):
    """
    Gets the required vector f given u, w, delta x and delta t
    """
    u_diff_approx = (np.roll(u, -1) - 2 * u + np.roll(u, 1))/(delx**2)
    w_diff_approx = (np.roll(w, -1) - 2 * w + np.roll(w, 1))/(delx**2)

    f = w + delt * u_diff_approx + ((delt/2)**2) * w_diff_approx
    return f

def compute_timesteps(M, delt, n, u_func, w_func):
    """
    Function that computes and plots a number of timesteps of the wave equation

    :param M: constant to determine grid width
    :param delt: delta t, the time difference between time steps
    :param n: number of time steps to be computed
    :param u_func: the actual function u(x,t) for comparison
    :param w_func: the actual function w(x,t) for comparison
    """ 
    # obtain grid width delta x
    delx = 1/M

    # obtain constant C1
    c = (delt / 2)**2
    c1 = c / (delx)**2

    # generate evenly distributed x values at grid width
    x_vals = (np.arange(M)+1) * delx
    # generate evenly distributed x values for the actual solution
    x_range = np.linspace(x_vals[0],1,501)

    # obtain the intial conditions
    u = u_func(x_vals, 0)
    w = w_func(x_vals, 0)

    for k in range(n):
        
        # obtain the w and u at the next time step
        f = get_f(u, w, delx, delt)
        new_w = solver_alg(c1, f)
        new_u = u + (delt / 2) * (w + new_w)

        # obtain the actual w and u from the solution at the next time step
        actual_w_vals = w_func(x_range, k*delt)
        actual_u_vals = u_func(x_range, k*delt) 

        # plot the discretisation and the actual function every plot_step timesteps
        plot_step = 50
        if k % plot_step == 0:
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(x_vals, np.real(new_u), color = color)
            plt.plot(x_range, actual_u_vals, ls=':', color = "black")

        # update u and w
        u = new_u
        w = new_w
    
    plt.title("Plot of u(x,t) against x for different timesteps.   M = " + str(M)+", Delta t = "+str(delt))
    plt.xlabel("x")
    plt.ylabel("u(x,t)") 

def w_func(x,t):
    """
    The example function w(x,t) from the report
    """
    w_vals = (np.sin(2 * np.pi * x) / (2 * np.pi)) * np.cos(2 * np.pi * t)
    return w_vals

def u_func(x,t):
    """
    The example function u(x,t) from the report
    """
    u_vals = (np.sin(2 * np.pi * x) / (2 * np.pi)) * (np.sin(2 * np.pi * t) / (2 * np.pi))
    return u_vals

def test2g(M, delt, n):
    compute_timesteps(M, delt, n, u_func, w_func)
    plt.show()

"""
This script obtains the results from Q2 of the report
[compare_algs(2000) and compare_algs(5000) are commented out as they take a long time to run]
"""
if __name__ == '__main__':
    """ Compare algorithms as required in 2f) """
    compare_algs(500)
    compare_algs(1000)
    # compare_algs(2000)
    # compare_algs(5000)

    """ Compute timesteps of the discretisation of the function from 2g)"""
    test2g(500, 1/1000, 251)
    test2g(10, 1/1000, 251)


import numpy as np
import matplotlib.pyplot as plt
from cla_utils.exercises5 import solve_R
from scipy.sparse import csgraph
from scipy.linalg import solve_triangular

def block_diag(A,B):
    """
    Function that takes square matrices or numbers A, B as input
    and returns the block diagonal matrix = diag(A,B)
    """
    if type(A) == np.ndarray:
        m = A.shape[0]
    else:
        m = 1
        
    if type(B) == np.ndarray:
        n = B.shape[0]
    else:
        n = 1
        
    # dimension of new matrix
    k = m + n
    
    # construct the block diagonal matrix
    C = np.zeros((k,k))
    C[:m,:m] = A
    C[m:,m:] = B
    
    return C

def extra_givens(Hk, Qhp, k):
    """
    Function that applies one extra Givens rotation for a GMRES iteration
    :inputs: Hk, Qh and k (iteration number)
    :outputs: the new Q and R
    """    
    if k == 0:
        Qh = np.eye(2)
    else:
        Qh = block_diag(Qhp, 1)
    
    A = Qh.T @ Hk
        
    theta = np.arctan(A[k+1,k] / A[k,k])
    c = np.cos(theta)
    s = np.sin(theta)
    M = np.array([[c, s],[-s, c]])
    
    A[k: k+2, :] = M @ A[k: k+2, :]
    Qhk = block_diag(np.eye(k), M)
    
    newR = A.copy()
    newQ = (Qhk @ Qh.T).T

    return newQ, newR

def GMRES(A, b, apply_pc, maxit, tol, x0=None, return_residual_norms=False, return_residuals=False):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param apply_pc: preconditioning function    
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param x0: the initial guess (if not present, use b)
    :param return_residual_norms: logical
    :param return_residuals: logical

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual
    at iteration k
    """
    m = np.size(b)

    # Obtain the new b from the preconditioning function
    b = apply_pc(b)
    
    if x0 is None:
        x0 = b
    
    # normalise the inital vector
    x = x0 / np.linalg.norm(x0)

    # count number of iterations
    k = 0

    # matrix storing the norms of the residuals at each iteration
    rnorms = np.zeros(maxit)
    # list storing the residuals at each iteration
    r = []

    # initialise the variables as required
    Qh = None
    Rh = None
    Q = np.zeros((m, maxit+1))
    H = np.zeros((maxit+1,maxit))
    Q[:,0] = x / np.linalg.norm(x)

    while True:
        ### Apply step n of Arnoldi

        # Solve Mv = AQk
        v = apply_pc(A @ Q[:,k])

        H[:k+1, k] = Q[:, :k+1].T.conjugate() @ v
        v = v - Q[:, :k+1] @ H[:k+1, k]

        H[k+1, k] = np.linalg.norm(v)
        Q[:, k+1] = v / np.linalg.norm(v)

        Hk = H[:(k+1)+1,:(k)+1]
        Qk = Q[:,:(k+1)]
        ### End of step n of Arnoldi

        # create basis vector e1
        e1 = np.eye((k+1)+1)[:,0]
        
        # Obtain the QR factorisation of Hk via Givens rotations
        Qh, Rh = extra_givens(Hk, Qh, k)

        # Find y by least squares
        Qh_reduced = Qh[:, :k+1]    
        Rh_reduced = Rh[:k+1, :k+1]

        # Back substitution
        y = solve_R(Rh_reduced, Qh_reduced.conjugate().T @ (np.linalg.norm(b) * e1))
        
        # Update the solution
        x = Qk @ y

        # Update the residuals and residual norms
        r.append(Hk @ y - np.linalg.norm(b) * e1)
        rnorms[k] = np.linalg.norm(r[k])

        # Increment the iteration counter
        k += 1

        # check convergence criteria
        R = rnorms[k-1]
        if R < tol:
            nits = k
            break
        elif k+1 > maxit:
            nits = -1
            break
    
    # Return the required variables as required by the logicals
    if return_residual_norms and return_residuals:
        return x, nits, rnorms[:nits+1], r
    elif return_residual_norms and not return_residuals:
        return x, nits, rnorms[:nits+1]
    elif return_residuals and not return_residual_norms:
        return x, nits, r
    else:
        return x, nits 

"""
for c in range(1, 10):
    M = c*U
    evals = np.linalg.eigvals(solve_triangular(M, A))
    #plt.scatter(np.real(evals),np.imag(evals))
    #plt.show()
    print(evals)
    print(np.abs(1-evals))
"""

def investigate_GMRES(L, c):
    m, _ = L.shape
    I = np.eye(m)
    A = I + L
    e = np.linalg.eigvals(A)
    #e = np.sort(e)
    plt.scatter(e,1+np.ones(len(e)),marker = 'x')
    plt.title("A evals")
    plt.show()

    U = np.triu(A)

    #c = 5
    M = c*U
    e = np.linalg.eigvals(M)

    evals = np.linalg.eigvals(solve_triangular(M, A))
    plt.scatter(np.real(evals),np.imag(evals), marker='x')
    plt.title("Minv A evals")
    plt.show()

    #print(evals)
    #print(np.abs(1-evals))
    c11 = np.max(np.abs(1-evals))
    c1 = np.linalg.norm(I - solve_triangular(M, A), ord = 2)

    b = np.random.randn(m)

    def apply_pc_M(x):
        x_tilde = solve_R(M, x)
        return x_tilde

    def apply_pc_I(x):
        return x

    x1, nits1, rrn1, rr1 = GMRES(A, b, apply_pc_I, maxit=1000, tol=1.0e-9, return_residual_norms=True, return_residuals=True)
    x2, nits2, rrn2, rr2 = GMRES(A, b, apply_pc_M, maxit=1000, tol=1.0e-9, return_residual_norms=True, return_residuals=True)

    k=rrn2[0]
    xvals = np.arange(len(rrn2))
    x_range = np.linspace(0,xvals[-1],200)
    plt.plot(xvals,rrn2, ls='--')
    #plt.plot(xvals,rrn1, ls='--')
    plt.plot(x_range, k*c1**(x_range))
    plt.show()

    plt.semilogy(rrn2)
    plt.semilogy(x_range, k*c1**(x_range))
    plt.show()

    err1=np.linalg.norm(A@x1 - b)
    err2=np.linalg.norm(A@x2 - b)

    print("err1 is : " + str(err1))
    print("err2 is : " + str(err2))

    print("c1 is   : " + str(c1))
    print("nits1 is: " + str(nits1))
    print("nits2 is: " + str(nits2))

    return


m = 100
np.random.seed(55)
#S = np.random.randn(m,m)
#S = S+S.T
#S = np.random.randint(0,m,(m,m))
#S = S+S.T
#S = np.diag(np.diag(S))+np.triu(S,1)+np.triu(S,1).T
#S = np.diag(np.arange(15)+1) + np.diag(3*np.arange(14),1)+ np.diag(3*np.arange(14),-1)
S = np.arange(1,m+1) * np.arange(1,m+1)[:, np.newaxis]
S = S / (m**2)

L = csgraph.laplacian(S)
investigate_GMRES(L, 1)
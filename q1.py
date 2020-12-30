import numpy as np

"Q1"
def triA(c, d, m):
    A = np.eye(m) * c
    D = np.diag(np.ones(m-1) * d, 1)
    A += D + D.T
    return A
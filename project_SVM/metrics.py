import numpy as np
from numpy.matlib import repmat

# helper functions for kernel computation

def innerproduct(X,Z=None):
    '''Takes X,Z and Returns an array equal to inner product
    When X,Z are 2d arrays, it gives inner product for each row'''
    if Z is None: # case when there is only one input (X)
        Z = X
    
    inn_prod = np.dot(X, Z.T)  # first dim is the sample index
    # until here 
    return inn_prod
def l2distance(X, Z=None, p=2):
    """
    Compute all pairwise distances between vectors in X and Z.

    Parameters
    ----------
    X : np.array
        shape: (m1, d1)
    Z : np.array
        shape: (m2, d2)

    Returns
    -------
    L_2 : np.array
        A matrix L_2 of shape (m1, m2).  Each entry in L_2 i,j represents the
        distance between row i in X and row j in Z.
    """
    if Z is None:
        Z = X
        
    m1, d1 = X.shape
    m2, d2 = Z.shape
    assert (d1==d2), "Dimensions of input vectors must match!"
    # Your code goes here ..
    X_2 = (X*X).sum(axis=1).reshape((m1,1))*np.ones(shape=(1,m2))
    Z_2 = (Z*Z).sum(axis=1).reshape((1,m2))*np.ones(shape=(m1,1))
    cross = -2*innerproduct(X, Z)

    # L_2 distance
    dist_xz = np.abs((X_2 + Z_2 + cross).round(decimals=6))
    return dist_xz
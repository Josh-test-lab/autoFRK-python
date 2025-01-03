"""
Title: helper.py
Editor: Hsu, Yao-Chih
Version: 1140103
Original version: autoFRK v1.4.3 (https://CRAN.R-project.org/package=autoFRK)
Reference: Tzeng S, Huang H, Wang W, Nychka D, Gillespie C (2021). _autoFRK: Automatic Fixed Rank Kriging_. R package version 1.4.3, https://CRAN.R-project.org/package=autoFRK.
"""

### import modele
import numpy as np

### function
def eigenDecomposeInDecreasingOrder(mat):
    """
    Internal function: eigen-decomposition in decreasing order.
    
    Parameters: 
    mat (numpy.ndarray): A matrix.

    Returns: 
    A dictionary with 'value' (eigenvalues) and 'vector' (eigenvectors).
    """
    value, vector = np.linalg.eigh(mat)
    #value, vector = np.linalg.eig(mat)
    obj_value = value[::-1]
    obj_vector = vector[:, ::-1]
    
    return {'value': obj_value, 'vector': obj_vector}

def calculateLogDeterminant(R, L, K):
    """
    Internal function: calculate the log determinant for likelihood use.

    Parameters:
    R (numpy.ndarray): A p x p positive-definite matrix.
    L (numpy.ndarray): A p x K matrix.
    K (int): A numeric.

    Returns:
    A numeric.
    """
    first_part_determinant = logDeterminant(np.eye(K) + L.T @ np.linalg.inv(R) @ L)
    second_part_determinant = logDeterminant(R)
    
    return first_part_determinant + second_part_determinant

def computeLikelihood(data, Fk, M, s, Depsilon):
    """
    Internal function: compute a negative log-likelihood (-2*log(likelihood))

    Parameters:
    data: An \emph{n} by \emph{T} data matrix (NA allowed) with \eqn{z[t]} as the \emph{t}-th column.
    Fk: An \emph{n} by \emph{K} matrix of basis function values with each column being a basis function taken values at \code{loc}.
    M: A \emph{K} by \emph{K} symmetric matrix.
    s: A scalar.
    Depsilon: An \emph{n} by \emph{n} diagonal matrix.
    
    Returns:
    A numeric.
    """
    data = np.asarray(data)
    non_missing_points_matrix = ~np.isnan(data)
    num_columns = data.shape[1]

    

    return n2loglik












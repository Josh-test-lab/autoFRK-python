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
def ncol(data):
    """
    Convert `ncol` in R to Python
    """
    data = np.asarray(data)
    return data.shape[1] if data.ndim != 1 else None

def NCOL(data):
    """
    Convert `NCOL` in R to Python
    """
    data = np.asarray(data)
    return data.shape[1] if data.ndim != 1 else 1

def eigenDecomposeInDecreasingOrder(mat):
    """
    Internal function: eigen-decomposition in decreasing order.
    
    Parameters: 
    mat (numpy.ndarray): A matrix.

    Returns: 
    A dictionary with 'value' (eigenvalues) and 'vector' (eigenvectors).
    """
    value, vector = np.linalg.eig(mat)
    increase_index = np.argsort(value)[::-1]
    obj_value = value[increase_index]
    obj_vector = vector[:, increase_index]
    
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
    data (np.ndarray): An n x T data matrix (NA allowed) with z[t] as the t-th column.
    Fk (np.ndarray): An n x K matrix of basis function values.
    M (np.ndarray): A K x K symmetric matrix.
    s (float): A scalar.
    Depsilon (np.ndarray): An n x n diagonal matrix.
    
    Returns:
    A numeric.
    """
    data = np.asarray(data)
    non_missing_points_matrix = ~np.isnan(data)
    num_columns = NCOL(data)

    n2loglik = np.sum(non_missing_points_matrix) * np.log(2 * np.pi)
    R = toSparseMatrix(s * Depsilon)
    eg = eigenDecompose(M)

    K = NCOL(Fk)
    L = Fk @ eg['vector'] @ np.diag(np.sqrt(np.maximum(eg['vector'], 0)), K) @ eg['vector'].T

    if num_columns == 1:
        zt = data[non_missing_points_matrix]
        Rt = np.diag(R[non_missing_points_matrix, non_missing_points_matrix], 0)
        Lt = L[non_missing_points_matrix, :]
        n2loglik += calculateLogDeterminant(Rt, Lt, K) + np.sum(zt * invCz(Rt, Lt, zt))
    else:
        for t in range(num_columns):
            zt = data[non_missing_points_matrix[:, t], t]
            Rt = R[np.ix_(non_missing_points_matrix[:, t], non_missing_points_matrix[:, t])]
            Lt = L[non_missing_points_matrix[:, t], :]
            n2loglik += calculateLogDeterminant(Rt, Lt, K) + np.sum(zt * invCz(Rt, Lt, zt))
        
    return n2loglik


















































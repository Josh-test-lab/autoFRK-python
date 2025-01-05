"""
Title: helper.py
Editor: Hsu, Yao-Chih
Version: 1140103
Original version: autoFRK v1.4.3 (https://CRAN.R-project.org/package=autoFRK)
Reference: Tzeng S, Huang H, Wang W, Nychka D, Gillespie C (2021). _autoFRK: Automatic Fixed Rank Kriging_. R package version 1.4.3, https://CRAN.R-project.org/package=autoFRK.
"""

### import modele
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

### function
def transfor_to_matrix(data):
    data = np.asarray(data)
    data = np.matrix(data) if data.ndim != 1 else np.matrix(data).T
    return data

def eigenDecomposeInDecreasingOrder(mat):
    """
    Internal function: eigen-decomposition in decreasing order.
    
    Parameters: 
    mat: A matrix.

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
    R: A p x p positive-definite matrix.
    L: A p x K matrix.
    K: A numeric.

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
    data: An n x T data matrix (NA allowed) with z[t] as the t-th column.
    Fk: An n x K matrix of basis function values.
    M: A K x K symmetric matrix.
    s: A scalar.
    Depsilon: An n x n diagonal matrix.
    
    Returns:
    A numeric.
    """
    data = transfor_to_matrix(data)
    non_missing_points_matrix = ~np.isnan(data)
    num_columns = data.shape[1]

    n2loglik = np.sum(non_missing_points_matrix) * np.log(2 * np.pi)
    R = toSparseMatrix(s * Depsilon)
    eg = eigenDecompose(M)

    FK = transfor_to_matrix(FK)
    K = Fk.shape[1]
    L = Fk @ eg['vector'] @ np.diag(np.sqrt(np.maximum(eg['vector'], 0)), K) @ eg['vector'].T

    for t in range(num_columns):
        zt = data[non_missing_points_matrix[:, t], t]
        Rt = R[np.ix_(non_missing_points_matrix[:, t], non_missing_points_matrix[:, t])]
        Lt = L[non_missing_points_matrix[:, t], :]
        n2loglik += calculateLogDeterminant(Rt, Lt, K) + np.sum(zt * invCz(Rt, Lt, zt))
        
    return n2loglik

def selectBasis(data, loc, D = None, maxit = 50, avgtol = 1e-6, max_rank = None, sequence_rank = None, method = ['fast', 'EM'], num_neighbors = 3, max_knot = 5000, DfromLK = None, Fk = None):
    """
    Internal function: select basis functions

    Parameters:
    data: An n x T data matrix (NA allowed) with z[t] as the t-th column.
    loc: n x d matrix of coordinates corresponding to n locations.
    D: A diagonal matrix.
    maxit: An integer for the maximum number of iterations used in indeMLE.
    avgtol: A numeric for average tolerance used in indeMLE.
    max_rank: An integer of the maximum of K values.
    sequence_rank: An array of K values
    method: A character of a list of characters.
    num_neighbors: An integer.
    max_knot: An integer for the maximum number of knots
    DfromLK: An n x n diagonal matrix.
    Fk: An n x K matrix of basis function values with each column being a basis function taken values at `loc`.
    
    Returns:
    An mrts object with 6 attributes
    """
    data = transfor_to_matrix(data)
    are_all_missing_in_columns = np.all(np.isnan(data), axis = 0)
    if np.any(are_all_missing_in_columns):
        data = np.matrix(data[:, ~are_all_missing_in_columns])
    if D is None:
        D = np.eye(data.shape[0])

    loc = transfor_to_matrix(loc)
    d = loc.shape[1]
    is_data_with_missing_values = np.any(np.isnan(data))
    na_rows = np.where(np.sum(~np.isnan(data), axis = 1) == 0)[0]
    pick = np.arange(data.shape[0])
    if len(na_rows) > 0:
        data = data[~np.isin(pick, na_rows), :]
        D = D[~np.isin(pick, na_rows), :][:, ~np.isin(pick, na_rows)]
        pick = pick[~np.isin(pick, na_rows)]
        is_data_with_missing_values = np.any(np.isnan(data))

    N = len(pick)
    klim = min(N, round(10 * np.sqrt(N)))
    if N < max_knot:
        knot = loc[pick, :]
    else:
        knot = subKnot(loc[pick, :], min(max_knot, klim))

    if max_rank is not None:
        max_rank = round(max_rank)
    else:
        max_rank = round(max(sequence_rank)) if sequence_rank is not None else klim

    if sequence_rank is not None:
        K = np.asarray(np.unique(np.round(sequence_rank).astype(int)))
        if max(K) > max_rank:
            raise ValueError('maximum of sequence_rank is larger than max_rank!')
        if (np.all(K <= d)):
            raise ValueError('Not valid sequence_rank!')
        elif (np.any(K < (d + 1))):
            print(f'\033[93mWarning: The minimum of sequence_rank can not less than {d + 1}. Too small values will be ignored\033[0m.')
        K = K[K > d]
    else:
        K = np.unique(np.round(np.arange(d + 1, max_rank + 1, max_rank ** (1 / 3) * d)))
        if len(K) > 30:
            K = np.unique(np.round(np.linspace(d + 1, max_rank, num = 30)).astype(int))

    if Fk is None:
        Fk = mrts(knot, max(K), loc, max_knot)
    AIC_list = np.full(len(K), np.inf)
    if method == 'fast' or method == ['fast', 'EM']:
        method = 'fast'
    elif method == 'EM':
        method = 'EM'
    else:
        raise ValueError('Invalid method. Choose from "fast" or "EM"!')
    num_data_columns = data.shape[1]

    if method == 'EM' and DfromLK is None:
        for k in range(len(K)):
            AIC_list[k] = indeMLE(
                                    data, 
                                    Fk[np.ix_(pick, np.arange(K[k]))], 
                                    D, 
                                    maxit, 
                                    avgtol, 
                                    wSave = False, 
                                    verbose = False
                                )['negloglik']
    else:
        if is_data_with_missing_values:
            for tt in range(num_data_columns):
                is_cell_missing_in_a_column = np.isnan(data[:, tt])
                if not np.any(is_cell_missing_in_a_column):
                    continue
                cidx = np.where(~is_cell_missing_in_a_column)[0]
                # 此處未驗證，未驗證起始
                nn = NearestNeighbors(n_neighbors = num_neighbors)
                nn.fit(loc[cidx, :])
                nnidx = nn.kneighbors(loc[is_cell_missing_in_a_column, :], return_distance = False)
                nnidx = cidx[nnidx]
                nnval = data[nnidx, tt]
                # 未驗證結束
                data[is_cell_missing_in_a_column, tt] = np.mean(nnval, axis = 1)

        if DfromLK is None:
            iD = np.linalg.inv(D)
            iDFk = iD @ Fk[pick, :]
            iDZ = iD @ data
        else:
            wX = DfromLK['wX'][pick, :]
            G = np.transpose(DfromLK['wX']) @ DfromLK['wX'] + DfromLK['lambda'] * DfromLK['Q']
            weight = DfromLK['weights'][pick]
            wwX = np.diag(np.sqrt(weight)) @ wX
            wXiG = wwX @ np.linalg.inv(G)
            iDFk = weight * Fk[pick, :] - wXiG @ (np.transpose(wwX) @ Fk[pick, :])
            iDZ = weight * data - wXiG @ (np.transpose(wwX) @ data)
        sample_covariance_trace = np.sum(iDZ * data) / num_data_columns
        for k, K_val in enumerate(K):
            inverse_square_root_matrix = getInverseSquareRootMatrix(Fk[pick, :K[k]], iDFk[:, :K[k]])
            ihFiD = inverse_square_root_matrix @ iDFk[:, :K[k]].T
            matrix_JSJ = np.dot(ihFiD @ data, (ihFiD @ data).T) / num_data_columns
            matrix_JSJ = (matrix_JSJ + matrix_JSJ.T) / 2
            AIC_list[k] = cMLE(
                                Fk = Fk[pick, :K[k]],
                                num_columns = num_data_columns,
                                sample_covariance_trace = sample_covariance_trace,
                                inverse_square_root_matrix = inverse_square_root_matrix,
                                matrix_JSJ = matrix_JSJ
                            )['negloglik']

    df = (K * (K + 1) / 2 + 1) * (K <= num_data_columns) + (K * num_data_columns + 1 - num_data_columns * (num_data_columns - 1) / 2) * (K > num_data_columns)
    AIC_list = AIC_list + 2 * df
    Kopt = K[np.argmin(AIC_list)]
    out = Fk[:, :Kopt]

    # 此處未驗證，未驗證起始
    attributes_out = {key: value for key, value in Fk.__dict__.items() if key != 'dim'}
    out.__dict__.update(attributes_out)
    # 未驗證結束

    return out

def estimateV(d, s, sample_covariance_trace, n):
    """
    Internal function: solve v parameter
    
    Parameters:
    d: An array of nonnegative values.
    s: A positive numeric.
    sample_covariance_trace: A positive numeric.
    n: An integer. Sample size.

    Returns:
    A numeric.
    """
    if np.max(d) < np.max(sample_covariance_trace / n, s):
        return np.max(sample_covariance_trace / n - s, 0)

    k = len(d)
    cumulative_d_values = np.cumsum(d)
    ks = np.arange(1, k + 1)
    if k == n:
        ks[n - 1] = n - 1
    eligible_indexes = np.where(d > (sample_covariance_trace - cumulative_d_values) / (n - ks))[0]
    L = np.max(eligible_indexes)
    if L >= n:
        L = n - 1
    
    return np.max((sample_covariance_trace - cumulative_d_values[L]) / (n - L) - s, 0)

def estimateEta(d, s, v):
    """
    Internal function: estimate eta parameter

    Parameters:
    d: An array of nonnegative values.
    s: A positive numeric.
    v: A positive numeric.

    Returns:
    A numeric.
    """
    return np.maximum(d - s - v, 0)

def neg2llik(d, s, v, sample_covariance_trace, sample_size):
    """
    Internal function: estimate negative log-likelihood

    Parameters:
    d: An array of nonnegative values.
    s: A positive numeric.
    v: A positive numeric.
    sample_covariance_trace: A positive numeric.
    sample_size: An integer. Sample size.

    Returns:
    A numeric.
    """
    k = len(d)
    eta = estimateEta(d, s, v)
    if np.max(eta / (s + v)) > 1e20:
        return np.inf
    else:
        return sample_size * np.log(2 * np.pi) + np.sum(np.log(eta + s + v)) + np.log(s + v) * (sample_size - k) + sample_covariance_trace / (s + v) - np.sum(d * eta / (eta + s + v)) / (s + v)
 
def computeNegativeLikelihood(nrow_Fk, ncol_Fk, s, p, matrix_JSJ, sample_covariance_trace, vfixed = None, ldet = 0):
    """
    Internal function: compute a negative likelihood
    
    Parameters:
    nrow_Fk: An integer. The number of rows of Fk.
    ncol_Fk: An integer. The number of columns of Fk.
    s: An integer.
    p: A positive integers. The number of columns of data.
    matrix_JSJ: A multiplication matrix
    sample_covariance_trace: A positive numeric.
    vfixed: A numeric
    ldet: A numeric. A log determinant.

    Returns:
    A list.
    """
    matrix_JSJ = transfor_to_matrix(matrix_JSJ)
    if not np.allclose(matrix_JSJ, matrix_JSJ.T):
        raise ValueError(f'Please input a symmetric matrix')
    if matrix_JSJ.shape[1] < ncol_Fk:
        raise ValueError(f'Please input the rank of a matrix larger than ncol_Fk = {ncol_Fk}')
    decomposed_JSJ = eigenDecomposeInDecreasingOrder(matrix_JSJ)
    eigenvalues_JSJ = decomposed_JSJ['value'][:ncol_Fk]
    eigenvectors_JSJ = transfor_to_matrix(decomposed_JSJ['vector'][:, :ncol_Fk])
    if vfixed is None:
        v = estimateV(eigenvalues_JSJ, s, sample_covariance_trace, nrow_Fk)
    else:
        v = vfixed
    d = np.maximum(eigenvalues_JSJ, 0)
    d_hat = estimateEta(d, s, v)
    negative_log_likelihood = (
        neg2llik(d, s, v, sample_covariance_trace, nrow_Fk) * p + ldet * p
    )

    return {
        "negative_log_likelihood": negative_log_likelihood,
        "P": eigenvectors_JSJ,
        "v": v,
        "d_hat": d_hat
    }

def compute_projection_matrix(Fk1, Fk2, data, S = None):
    """
    Internal function: maximum likelihood estimate with the likelihood.

    Parameters:
    Fk1: An n x K matrix.
    Fk2: An n x K matrix.
    data: An n x T data matrix.
    S: An n x n matrix. Default is None.

    Returns:
    A list.
    """
    num_columns = data.shape[1]
    inverse_square_root_matrix = getInverseSquareRootMatrix(Fk1, Fk2)
    inverse_square_root_on_Fk2 = inverse_square_root_matrix @ Fk2.T
    if S is None:
        matrix_JSJ = np.dot((inverse_square_root_on_Fk2 @ data), (inverse_square_root_on_Fk2 @ data).T) / num_columns
    else:
        matrix_JSJ = (inverse_square_root_on_Fk2 @ S) @ inverse_square_root_on_Fk2.T
    matrix_JSJ = (matrix_JSJ + matrix_JSJ.T) / 2
    
    return {
        "inverse_square_root_matrix": inverse_square_root_matrix,
        "matrix_JSJ": matrix_JSJ
    }

def isDiagonal(obj):
    """
    Internal function: check if a numeric-like object is diagonal.

    Parameters:
    obj: A numeric-like object.

    Returns:
    logical
    """
    if isinstance(obj, (int, float)) and np.isscalar(obj):
        return True
    if isinstance(obj, np.ndarray):
        if obj.shape[0] != obj.shape[1]:
            return False
        return np.sum(np.abs(np.diag(np.diag(obj)) - obj)) < np.finfo(float).eps
    
    return False

def subKnot(x, nknot, xrng = None, nsamp = 1):
    """
    Internal function: sampling knots
    
    Parameters:
    x: A matrix or an array
    nknot: A location matrix
    xrng: An array including two integers
    
    Returns:
    sampling knots
    """
    x = np.sort(x, axis = 0)
    xdim = x.shape
    
    if xrng is None:
        xrng = np.matrix([np.min(x, axis = 0).A.flatten(), np.max(x, axis = 0).A.flatten()])
    # To-do: move out the function based on SRP
    def mysamp(z_and_id):
        z = np.array(list(z_and_id.keys()), dtype = float)
        if len(z) == 1:
            return z
        else:
            np.random.seed(int(np.mean(list(z_and_id.values()))))
            return np.random.choice(z, size = min(nsamp, len(z)), replace = False)
    
    rng = np.sqrt(xrng[1, :] - xrng[0, :])
    rng[rng == 0] = min(rng[rng > 0]) / 5
    rng = rng * 10 / min(rng)
    rng_max_index = np.argmax(rng)
    
    nmbin = np.round(np.exp(np.log(rng) * np.log(nknot) / np.sum(np.log(rng))))
    nmbin = np.maximum(2, nmbin)
    while np.prod(nmbin) < nknot:
        nmbin[rng_max_index] += 1
    
    gvec = transfor_to_matrix(np.ones(xdim[0], dtype = int))
    cnt = 0
    while len(np.unique(gvec)) < nknot:
        nmbin += cnt
        kconst = 1
        gvec = transfor_to_matrix(np.ones(xdim[0], dtype = int))
        for kk in range(xdim[1]):
            grp = np.minimum(np.floor((nmbin[kk] - 1) * ((x[:, kk] - xrng[0, kk]) / (xrng[1, kk] - xrng[0, kk]))), nmbin[kk] - 1)
            if len(np.unique(grp)) < nmbin[kk]:
                brk = np.quantile(x[:, kk], np.linspace(0, 1, nmbin[kk] + 1))
                brk[0] -= 0.1**8
                grp = np.digitize(x[:, kk], brk)
            gvec += kconst * grp
            kconst *= nmbin[kk]
        cnt += 1
    # To-do: refactor the following four lines
    gvec = pd.Categorical(gvec)
    gid = pd.Series(gvec.codes).astype(float)
    gid.index = np.arange(1, xdim[0] + 1)
    index = np.concatenate([mysamp(group) for group in gid.groupby(gvec).groups.values()])
    
    return x[index.astype(int), :]

def fetchNonZeroIndexs(mat):
    """
    Internal function: convert to a sparse matrix

    Parameters:
    mat: A matrix.
    
    Returns:
    An array of indeces
    """
    if not isinstance(mat, (np.ndarray, csr_matrix)):
        raise TypeError(f'Wrong matrix format, but got {type(mat)}')
    if isinstance(mat, csr_matrix):
        row_indices, col_indices = mat.nonzero()
    else:
        row_indices, col_indices = np.nonzero(mat)

    row_indices, col_indices = np.nonzero(mat)
    sorted_col_indices = np.argsort(col_indices)
    nr, nc = mat.shape
    linear_indices = (col_indices[sorted_col_indices]) * nr + row_indices[sorted_col_indices]
    
    return linear_indices

def toSparseMatrix(mat, verbose = FALSE):
    """
    Internal function: convert to a sparse matrix

    Parameters:
    mat: A matrix or a dataframe
    verbose: A boolean

    Returns:
    sparse matrix
    """



















































































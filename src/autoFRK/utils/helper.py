"""
Title: Some helpful functions of autoFRK-Python Project
Author: Yao-Chih Hsu
Version: 1141025
Description: This file contains some helper functions used in the autoFRK-Python project.
Reference: `autoFRK` R package by Wen-Ting Wang from https://github.com/egpivo/autoFRK
"""

# import modules
import torch
import numpy as np
from scipy.integrate import quad
from typing import Callable, Dict, Union
from ..utils.logger import LOGGER
from ..utils.utils import to_tensor
from ..utils.matrix_operator import getInverseSquareRootMatrix, invCz

# convert dense tensor to sparse matrix, using in indeMLE
# python 不需要，在 R 中僅作為節省記憶體的角色
# def toSparseMatrix(
#     mat: torch.Tensor, 
#     verbose: bool=False
# ) -> torch.Tensor:
#     if not torch.is_tensor(mat):
#         warn_msg = f'Expected tensor, but got {type(mat)}'
#         LOGGER.warning(warn_msg)
#         mat = torch.tensor(mat)
#     if mat.is_sparse:
#         if verbose:
#             info_msg = f'The input is already a sparse tensor'
#             LOGGER.info(info_msg)
#         return mat
#     if verbose:
#         return mat.to_sparse()

# fast mode KNN for missing data imputation, using in autoFRK
# Its have OpenMP issue, set environment variable OMP_NUM_THREADS=1 to avoid it, or use sklearn version below
# check = ok
def fast_mode_knn_faiss(
    data: torch.Tensor,
    loc: torch.Tensor, 
    n_neighbor: int = 3
) -> torch.Tensor:
    """
    Impute missing values in data using fast KNN with Faiss.

    This function performs nearest-neighbor imputation for missing values in
    spatial data. For each time point (column), missing entries are replaced
    by the mean of their k nearest neighbors, computed via Faiss.

    Parameters
    ----------
    data : torch.Tensor
        Input data of shape (N, T), where N is the number of spatial locations
        and T is the number of time points. Missing values should be NaN.
    loc : torch.Tensor
        Coordinates of spatial locations, shape (N, spatial_dim).
    n_neighbor : int, optional
        Number of nearest neighbors to use for imputation. Default is 3.

    Returns
    -------
    torch.Tensor
        Data tensor with missing values imputed, same shape and dtype/device
        as the input.
    """
    import faiss

    dtype=data.dtype
    device=data.device

    data = data.detach().cpu().numpy()
    loc = loc.detach().cpu().numpy()

    # use faiss on GPU if available
    if device.type != 'cpu':
        res = faiss.StandardGpuResources()

    for tt in range(data.shape[1]):
        col = data[:, tt]
        where = np.isnan(col)
        if not np.any(where):
            continue

        known_idx = np.where(~where)[0]
        unknown_idx = np.where(where)[0]

        # if low known values
        if len(known_idx) < n_neighbor:
            err_msg = f'Column {tt} has too few known values to impute ({len(known_idx)} < {n_neighbor}).'
            LOGGER.warning(err_msg)
            raise ValueError(err_msg)

        # use faiss for KNN
        index = faiss.IndexFlatL2(loc.shape[1])
        if device.type != 'cpu':
            index = faiss.index_cpu_to_gpu(res, 0, index)

        # get the values of neighbors
        index.add(loc[known_idx])
        _, knn_idx = index.search(loc[unknown_idx], n_neighbor)

        # impute missing values with the mean of neighbors
        neighbor_vals = col[known_idx[knn_idx]]
        col[where] = np.nanmean(neighbor_vals, axis=1)
        data[:, tt] = col

    return torch.tensor(data, dtype=dtype, device=device)

# fast mode KNN for missing data imputation, using in autoFRK, sklearn version
# check = ok
def fast_mode_knn_sklearn(
    data: torch.Tensor,
    loc: torch.Tensor,
    n_neighbor: int = 3
) -> torch.Tensor:
    """
    Impute missing values in data using fast KNN with scikit-learn.

    This function performs nearest-neighbor imputation for missing values in
    spatial data. For each time point (column), missing entries are replaced
    by the mean of their k nearest neighbors, computed via scikit-learn's
    NearestNeighbors.

    Parameters
    ----------
    data : torch.Tensor
        Input data of shape (N, T), where N is the number of spatial locations
        and T is the number of time points. Missing values should be NaN.
    loc : torch.Tensor
        Coordinates of spatial locations, shape (N, spatial_dim).
    n_neighbor : int, optional
        Number of nearest neighbors to use for imputation. Default is 3.

    Returns
    -------
    torch.Tensor
        Data tensor with missing values imputed, same shape and dtype/device
        as the input.
    """
    from sklearn.neighbors import NearestNeighbors

    dtype = data.dtype
    device = data.device

    data = data.detach().cpu().numpy()
    loc = loc.detach().cpu().numpy()

    for tt in range(data.shape[1]):
        col = data[:, tt]
        where = np.isnan(col)
        if not np.any(where):
            continue

        known_idx = np.where(~where)[0]
        unknown_idx = np.where(where)[0]

        if 0 < len(known_idx) < n_neighbor:
            err_msg = f'Column {tt} has too few known values to impute ({len(known_idx)} < {n_neighbor}).'
            LOGGER.warning(err_msg)
            raise ValueError(err_msg)

        knn = NearestNeighbors(n_neighbors=n_neighbor, algorithm='auto').fit(loc[known_idx])
        distances, knn_idx = knn.kneighbors(loc[unknown_idx])

        neighbor_vals = col[known_idx[knn_idx]]
        col[where] = np.nanmean(neighbor_vals, axis=1)
        data[:, tt] = col

    return torch.tensor(data, dtype=dtype, device=device)

# fast mode KNN for missing data imputation, using in autoFRK, torch version
def fast_mode_knn_torch(
    data: torch.Tensor,
    loc: torch.Tensor,
    n_neighbor: int = 3
) -> torch.Tensor:
    """
    Impute missing values in data using fast KNN with PyTorch.

    Parameters
    ----------
    data : torch.Tensor
        (N, T) tensor with NaNs indicating missing values.
    loc : torch.Tensor
        (N, D) coordinates of spatial points.
    n_neighbor : int, optional
        Number of nearest neighbors used for imputation. Default = 3.

    Returns
    -------
    torch.Tensor
        Same shape as input, with missing values imputed.
    """
    imputed = data.clone()

    for tt in range(data.shape[1]):
        col = imputed[:, tt]
        mask = torch.isnan(col)

        if not mask.any():
            continue

        known_idx = (~mask).nonzero(as_tuple=True)[0]
        unknown_idx = mask.nonzero(as_tuple=True)[0]

        if 0 < len(known_idx) < n_neighbor:
            err_msg = f'Column {tt} has too few known values to impute ({len(known_idx)} < {n_neighbor}).'
            LOGGER.warning(err_msg)
            raise ValueError(err_msg)

        dist_known = torch.cdist(loc[unknown_idx], loc[known_idx])
        knn_idx_local = dist_known.topk(k = n_neighbor, largest=False).indices
        neighbor_vals = col[known_idx][knn_idx_local]
        col_clone = col.clone()
        col_clone[unknown_idx] = torch.nanmean(neighbor_vals, dim=1)
        imputed[:, tt] = col_clone

    return imputed

# select basis function for autoFRK, using in autoFRK
# check = none
def selectBasis(
    data: torch.Tensor,
    loc: torch.Tensor,
    D: torch.Tensor = None,
    maxit: int = 50,
    avgtol: float = 1e-6,
    max_rank: int = None,
    sequence_rank: torch.Tensor = None,
    method: str = "fast",
    num_neighbors: int = 3,
    max_knot: int = 5000,
    DfromLK: dict = None,
    Fk: dict = None,
    tps_method: str = "rectangular",
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Selecting basis functions in autoFRK.

    This function selects the optimal set of multi-resolution thin-plate spline (MRTS)
    basis functions for a given dataset, optionally handling missing values
    using fast KNN imputation or EM-based optimization. It evaluates candidate
    numbers of basis functions (K) using a negative log-likelihood criterion.

    Parameters
    ----------
    data : torch.Tensor
        An (n, T) data matrix where each column corresponds to a time point.
        Missing values (NaN) are allowed.
    loc : torch.Tensor
        An (n, d) matrix of spatial coordinates for the n observations.
    D : torch.Tensor, optional
        Diagonal matrix for measurement error covariance. Defaults to identity.
    maxit : int, optional
        Maximum number of iterations for indeMLE optimization. Default is 50.
    avgtol : float, optional
        Convergence tolerance for indeMLE. Default is 1e-6.
    max_rank : int, optional
        Maximum number of basis functions. Default is computed from sequence_rank or data size.
    sequence_rank : torch.Tensor, optional
        Candidate numbers of basis functions to test.
    method : str, optional
        Method for estimation. Options:
            - "fast": approximate KNN imputation using PyTorch (default)
            - "EM": expectation-maximization
    num_neighbors : int, optional
        Number of neighbors for KNN imputation. Default is 3.
    max_knot : int, optional
        Maximum number of knots to use when generating basis functions. Default is 5000.
    DfromLK : dict, optional
        Precomputed fine-scale components for LatticeKrig-style modeling.
    Fk : dict, optional
        Precomputed basis function values. If None, computed internally.
    tps_method : str, optional
        Specifies the method used to compute thin-plate splines (TPS). Default is None.
        Options:
            - "rectangular": Compute TPS in Euclidean (rectangular) coordinates.
            - "spherical_fast": Use spherical coordinates but apply the rectangular TPS formulation for faster computation.
            - "spherical": Compute TPS directly in spherical coordinates.
    dtype : torch.dtype, optional
        Data type for computations. Default is torch.float64.
    device : torch.device or str, optional
        Target device for computations ("cpu" or "cuda"). Default is 'cpu'.

    Returns
    -------
    dict
        A dictionary containing:
        - **MRTS** : (n, k) tensor of basis function values at the evaluation locations
        - **UZ** : transformed matrix for internal computation (if available)
        - **Xu** : (n, d) tensor of unique knots used
        - **nconst** : normalization constants for each basis function
        - **BBBH** : (optional) projection matrix times Phi
    """
    from ..utils.estimator import indeMLE, cMLE

    not_all_nan = ~torch.isnan(data).all(dim=0)
    data = data[:, not_all_nan]
    is_data_with_missing_values = torch.isnan(data).any()

    na_rows = torch.isnan(data).all(dim=1)
    pick = torch.arange(data.shape[0], dtype=torch.int64, device=device)

    if D is None:
        D = torch.eye(data.shape[0], dtype=dtype, device=device)

    if na_rows.any():
        data = data[~na_rows]
        D = D[~na_rows][:, ~na_rows]
        pick = pick[~na_rows]
        is_data_with_missing_values = torch.isnan(data).any()

    d = loc.shape[1]
    N = len(pick)
    klim = int(min(N, np.round(10 * np.sqrt(N))))
    if N < max_knot:
        knot = loc[pick, :]
    else:
        knot = subKnot(x        = loc[pick, :],
                       nknot    = min(max_knot, klim),
                       xrng     = None,
                       nsamp    = 1,
                       dtype    = dtype,
                       device   = device
                       )

    if max_rank is not None:
        max_rank = torch.round(max_rank)
    else:
        max_rank = torch.round(torch.max(sequence_rank)).to(torch.int) if sequence_rank is not None else klim

    if sequence_rank is not None:
        K = torch.unique(torch.round(sequence_rank).to(torch.int))
        if K.max() > max_rank:
            err_msg = f'maximum of sequence_rank is larger than max_rank!'
            LOGGER.error(err_msg)
            raise ValueError(err_msg)
        elif torch.all(K <= d):
            err_msg = f'Not valid sequence_rank!'
            LOGGER.error(err_msg)
            raise ValueError(err_msg)
        elif torch.any(K < (d + 1)):
            warn_msg = f'The minimum of sequence_rank can not less than {d + 1}. Too small values will be ignored.'
            LOGGER.warning(warn_msg)
        K = K[K > d]
    else:
        step = max_rank ** (1/3) * d
        K = torch.arange(d + 1, max_rank, step, dtype=dtype, device=device).round().to(torch.int).unique()
        if len(K) > 30:
            K = torch.linspace(d + 1, max_rank, 30, dtype=dtype, device=device).round().to(torch.int).unique()

    if Fk is None:
        from ..mrts import MRTS
        mrts = MRTS(dtype   = dtype,
                    device  = device
                    )
        Fk = mrts.forward(knot      = knot,
                          k         = max(K),
                          x         = loc,
                          maxknot   = max_knot,
                          tps_method= tps_method,
                          dtype     = dtype,
                          device    = device
                          )

    AIC_list = to_tensor([float('inf')] * len(K), dtype=dtype, device=device)
    num_data_columns = data.shape[1]

    if method == "EM" and DfromLK is None:
        for k in range(len(K)):
            AIC_list[k] = indeMLE(data  = data,
                                  Fk    = Fk["MRTS"][pick, :K[k]],
                                  D     = D,
                                  maxit = maxit,
                                  avgtol= avgtol,
                                  wSave = False,
                                  DfromLK= None,
                                  vfixed = None,
                                  verbose= False,
                                  dtype  = dtype,
                                  device = device
                                  )["negloglik"]

    else:
        if is_data_with_missing_values:
            if method == "fast":
                data = fast_mode_knn_torch(data       = data,
                                           loc        = loc, 
                                           n_neighbor = num_neighbors
                                           )
            # methods quited to use
            #elif method == "fast_sklearn":
            #    data = fast_mode_knn_sklearn(data       = data,
            #                                 loc        = loc, 
            #                                 n_neighbor = num_neighbors
            #                                 )
            #elif method == "fast_faiss":  # have OpenMP issue
            #    data = fast_mode_knn_faiss(data         = data,
            #                               loc          = loc, 
            #                               n_neighbor   = num_neighbors
            #                               )
        if DfromLK is None:
            iD = torch.linalg.solve(D, torch.eye(D.shape[0], dtype=dtype, device=device))
            iDFk = iD @ Fk["MRTS"][pick, :]
            iDZ = iD @ data
        else:
            wX = DfromLK["wX"][pick, :]
            G = DfromLK["wX"].T @ DfromLK["wX"] + DfromLK["lambda"] * DfromLK["Q"]
            weight = DfromLK["weights"][pick]
            wwX = torch.diag(torch.sqrt(weight)) @ wX
            wXiG = torch.linalg.solve(G, wwX.T).T
            iDFk = weight * Fk["MRTS"][pick, :] - wXiG @ (wwX.T @ Fk["MRTS"][pick, :])
            iDZ = weight * data - wXiG @ (wwX.T @ data)

        sample_covariance_trace = torch.sum(iDZ * data) / num_data_columns

        for k in range(len(K)):
            Fk_k = Fk["MRTS"][pick, :K[k]]
            iDFk_k = iDFk[:, :K[k]]
            inverse_square_root_matrix = getInverseSquareRootMatrix(left_matrix  = Fk_k,
                                                                    right_matrix = iDFk_k
                                                                    )
            ihFiD = inverse_square_root_matrix @ iDFk_k.T
            tmp = torch.matmul(ihFiD, data)
            matrix_JSJ = torch.matmul(tmp, tmp.T) / num_data_columns
            matrix_JSJ = (matrix_JSJ + matrix_JSJ.T) / 2
            AIC_list[k] = cMLE(Fk                           = Fk_k,
                               num_columns                  = num_data_columns,
                               sample_covariance_trace      = sample_covariance_trace,
                               inverse_square_root_matrix   = inverse_square_root_matrix,
                               matrix_JSJ                   = matrix_JSJ,
                               s                            =  0,
                               ldet                         =  0,
                               wSave                        =  False,
                               onlylogLike                  =  None,
                               vfixed                       =  None,
                               dtype                        = dtype,
                               device                       = device
                               )["negloglik"]

    df = torch.where(
        K <= num_data_columns,
        (K * (K + 1) / 2 + 1),
        (K * num_data_columns + 1 - num_data_columns * (num_data_columns - 1) / 2)
    )

    AIC_list = AIC_list + 2 * df
    Kopt = K[torch.argmin(AIC_list)].item()
    Fk["MRTS"] = Fk["MRTS"][:, :Kopt]
    return Fk

# subset knot selection for autoFRK, using in selectBasis
# check = none
def subKnot(
    x: torch.Tensor, 
    nknot: int, 
    xrng: torch.Tensor = None, 
    nsamp: int = 1, 
    dtype: torch.dtype=torch.float64,
    device: Union[torch.device, str]='cpu'
) -> torch.Tensor:
    """
    Sample knots for multi-resolution thin-plate splines (MRTS).

    This function selects a subset of spatial locations ("knots") from the input
    dataset, ensuring coverage across each dimension. It is used in constructing
    MRTS basis functions when the number of available locations exceeds the desired
    number of knots.

    Parameters
    ----------
    x : torch.Tensor
        Input locations (n x d) where n is the number of samples and d is the dimension.
    nknot : int
        Number of knots to sample.
    xrng : torch.Tensor, optional
        Optional array (2 x d) specifying the min and max range for each dimension.
        If None, the range is computed from the data.
    nsamp : int, optional
        Number of points to sample per bin when stratifying the data. Default is 1.
    dtype : torch.dtype, optional
        Data type for computations. Default is torch.float64.
    device : torch.device or str, optional
        Target device for computations ("cpu" or "cuda"). Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        Subset of input locations selected as knots (nknot x d).
    """
    x = torch.sort(x, dim=0).values
    xdim = x.shape

    if xrng is None:
        xrng = torch.stack([x.min(dim=0).values, x.max(dim=0).values], dim=0)

    rng = torch.sqrt(xrng[1] - xrng[0])
    if (rng == 0).any():
        rng[rng == 0] = rng[rng > 0].min() / 5
    rng = rng * 10 / rng.min()
    rng_max_index = torch.argmax(rng).item()

    log_rng = torch.log(rng)
    nmbin = torch.round(torch.exp(log_rng * torch.log(to_tensor(nknot, dtype=dtype, device=device)) / log_rng.sum())).int()
    nmbin = torch.clamp(nmbin, min=2)

    while torch.prod(nmbin).item() < nknot:
        nmbin[rng_max_index] += 1

    gvec = torch.ones(xdim[0], dtype=torch.int64, device=device)
    cnt = 0
    while len(torch.unique(gvec)) < nknot:
        nmbin += cnt
        kconst = 1
        gvec = torch.ones(xdim[0], dtype=torch.int64, device=device)
        for kk in range(xdim[1]):
            delta = xrng[1, kk] - xrng[0, kk]
            if delta == 0:
                grp = torch.zeros(xdim[0], dtype=torch.int64, device=device)
            else:
                grp = ((nmbin[kk] - 1) * (x[:, kk] - xrng[0, kk]) / delta).round().int()
                grp = torch.clamp(grp, max=nmbin[kk] - 1)

            if len(torch.unique(grp)) < nmbin[kk]:
                brk = torch.quantile(x[:, kk], torch.linspace(0, 1, nmbin[kk] + 1, dtype=dtype, device=device))
                brk[0] -= 1e-8
                grp = torch.bucketize(x[:, kk], brk) - 1
            gvec += kconst * grp
            kconst = kconst * nmbin[kk]

        cnt += 1

    # To-do: refactor the following lines
    # outside
    # need fix
    unique_g, inverse = torch.unique(gvec, return_inverse=True)
    mask = torch.zeros(xdim[0], dtype=torch.bool, device=device)
    for i, cnt in enumerate(torch.bincount(inverse)):
        idx = torch.nonzero(inverse == i, as_tuple=True)[0]
        if cnt <= nsamp:
            mask[idx] = True
        else:
            torch.manual_seed(int(idx.float().mean().item()))
            perm = torch.randperm(cnt, device=idx.device)
            mask[idx[perm[:nsamp]]] = True

    index = torch.nonzero(mask, as_tuple=True)[0].to(dtype=torch.int64, device=device)
    return x[index]

# compute negative log likelihood for autoFRK, using in cMLE
# check = ok
def computeNegativeLikelihood(
    nrow_Fk: int,
    ncol_Fk: int,
    s: int,
    p: int,
    matrix_JSJ: torch.Tensor,
    sample_covariance_trace: float,
    vfixed: float = None,
    ldet: float = 0.0,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Compute the negative log-likelihood for a given basis matrix and covariance information.

    This function is used internally in MRTS/FRK computations to evaluate the
    fit of a model, optionally estimating the noise variance if not fixed. 
    Eigen decomposition is used to compute principal components and corresponding
    negative log-likelihood.

    Parameters
    ----------
    nrow_Fk : int
        Number of rows of the basis function matrix Fk (usually the number of observations).
    ncol_Fk : int
        Number of basis functions (columns of Fk) to use in the likelihood.
    s : int
        Effective sample size parameter for the model.
    p : int
        Number of variables or data columns (e.g., spatial locations).
    matrix_JSJ : torch.Tensor
        Symmetric covariance-like matrix (shape [nrow_Fk, nrow_Fk]).
    sample_covariance_trace : float
        Trace of the sample covariance matrix.
    vfixed : float, optional
        Fixed noise variance. If None, the variance is estimated from data.
    ldet : float, optional
        Log-determinant adjustment for the likelihood. Default is 0.0.
    dtype : torch.dtype, optional
        Tensor data type for computations. Default is torch.float64.
    device : torch.device or str, optional
        Device to perform computations ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    dict
        A dictionary containing:
        - 'negative_log_likelihood' : float
            The computed negative log-likelihood value.
        - 'P' : torch.Tensor
            Eigenvectors of matrix_JSJ corresponding to largest eigenvalues.
        - 'v' : float
            Noise variance (estimated or fixed).
        - 'd_hat' : torch.Tensor
            Estimated eigenvalues adjusted by noise variance.
    
    Raises
    ------
    ValueError
        If matrix_JSJ is not symmetric or its rank is insufficient.
    """
    if not torch.allclose(matrix_JSJ, matrix_JSJ.T, atol=1e-10):
        err_msg = f'Please input a symmetric matrix'
        LOGGER.error(err_msg)
        raise ValueError(err_msg)

    if matrix_JSJ.size(1) < ncol_Fk:
        err_msg = f'Please input the rank of a matrix larger than ncol_Fk = {ncol_Fk}'
        LOGGER.error(err_msg)
        raise ValueError(err_msg)

    eigenvalues_JSJ, eigenvectors_JSJ = torch.linalg.eigh(matrix_JSJ)
    idx = torch.argsort(eigenvalues_JSJ, descending=True)
    eigenvalues_JSJ = eigenvalues_JSJ[idx][:ncol_Fk]
    eigenvectors_JSJ = eigenvectors_JSJ[:, idx][:, :ncol_Fk]

    if vfixed is None:
        v = estimateV(d                         = eigenvalues_JSJ, 
                      s                         = s, 
                      sample_covariance_trace   = sample_covariance_trace, 
                      n                         = nrow_Fk,
                      dtype                     = dtype,
                      device                    = device
                      )
    else:
        v = vfixed

    d = torch.clamp(eigenvalues_JSJ, min=0)
    d_hat = estimateEta(d = d,
                        s = s,
                        v = v
                        )

    negative_log_likelihood = neg2llik(d                        = d, 
                                       s                        = s, 
                                       v                        = v, 
                                       sample_covariance_trace  = sample_covariance_trace, 
                                       sample_size              = nrow_Fk,
                                       dtype                    = dtype,
                                       device                   = device
                                       ) * p + ldet * p

    return {"negative_log_likelihood": negative_log_likelihood,
            "P": eigenvectors_JSJ,
            "v": v,
            "d_hat": d_hat
            }

# estimate the eta parameter for negative likelihood, using in computeNegativeLikelihood
# check = ok
def estimateV(
    d: torch.Tensor, 
    s: float, 
    sample_covariance_trace: float, 
    n: int,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> float:
    """
    Estimate the parameter v.

    Parameters
    ----------
    d : torch.Tensor
        1D tensor of nonnegative eigenvalues (length k).
    s : float
        Positive numeric constant representing a lower bound adjustment.
    sample_covariance_trace : float
        Trace of the sample covariance matrix.
    n : int
        Total sample size.
    dtype : torch.dtype, optional
        Data type for intermediate tensor computations. Default is torch.float64.
    device : torch.device or str, optional
        Device for tensor computations ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    float
        Estimated parameter v.

    Raises
    ------
    ValueError
        If no eligible eigenvalues are found for the computation, indicating
        that inputs d, sample_covariance_trace, or n may be inconsistent.
    """
    if torch.max(d) < max(sample_covariance_trace / n, s):
        return max(sample_covariance_trace / n - s, 0.0)

    k = d.shape[0]
    cumulative_d_values = torch.cumsum(d, dim=0)
    ks = torch.arange(1, k + 1, dtype=dtype, device=device)
    if k == n:
        ks[-1] = n - 1

    eligible_indices = torch.nonzero(d > (sample_covariance_trace - cumulative_d_values) / (n - ks)).flatten()
    
    if len(eligible_indices) == 0:
        error_msg = "No eligible indices found: check input d, sample_covariance_trace, and n."
        LOGGER.error(error_msg)
        raise ValueError(error_msg)
    L = int(torch.max(eligible_indices))

    if (L + 1) >= n:
        L = n - 1
        v_hat = max((sample_covariance_trace - cumulative_d_values[L - 1]) / (n - L) - s, 0.0)
    else:
        v_hat = max((sample_covariance_trace - cumulative_d_values[L]) / (n - L - 1) - s, 0.0)
    return v_hat

# estimate the eta parameter for negative likelihood, using in computeNegativeLikelihood
# check = ok
def estimateEta(
    d: torch.Tensor, 
    s: float, 
    v: float
) -> torch.Tensor:
    """
    Estimate the eta parameter.

    Parameters
    ----------
    d : torch.Tensor
        1D tensor of nonnegative eigenvalues.
    s : float
        Positive numeric adjustment factor.
    v : float
        Positive numeric noise variance.

    Returns
    -------
    torch.Tensor
        Tensor of estimated eta values, same shape as input d.
    """
    return torch.clamp(d - s - v, min=0.0)

# compute the negative log likelihood, using in computeNegativeLikelihood
# check = ok
def neg2llik(
    d: torch.Tensor,
    s: float,
    v: float,
    sample_covariance_trace: float,
    sample_size: int,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> float:
    """
    Estimate the negative log-likelihood.

    Parameters
    ----------
    d : torch.Tensor
        1D tensor of nonnegative eigenvalues.
    s : float
        Positive numeric adjustment.
    v : float
        Positive noise variance.
    sample_covariance_trace : float
        Trace of the sample covariance matrix.
    sample_size : int
        Number of samples.
    dtype : torch.dtype, optional
        Desired torch dtype for computation (default: torch.float64).
    device : torch.device or str, optional
        Target device for computation (default: 'cpu').

    Returns
    -------
    float
        Estimated negative log-likelihood value.
    """
    k = d.shape[0]
    eta = estimateEta(d = d,
                      s = s,
                      v = v
                      )

    if torch.max(eta / (s + v)) > 1e20:
        return float("inf")
    s_plus_v = torch.as_tensor(s + v, device=device, dtype=dtype)
    log_det_term = torch.sum(torch.log(eta + s_plus_v))
    log_sv_term = torch.log(s_plus_v) * (sample_size - k)
    trace_term = sample_covariance_trace / (s_plus_v)
    eta_term = torch.sum(d * eta / (eta + s_plus_v)) / (s_plus_v)

    return sample_size * torch.log(torch.tensor(2 * torch.pi, device=device, dtype=dtype)) + log_det_term + log_sv_term + trace_term - eta_term

# using in EM0miss
# check = ok
def computeLikelihood(
    data: torch.Tensor,
    Fk: torch.Tensor,
    M: torch.Tensor,
    s: float,
    Depsilon: torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> float:
    """
    Compute the negative log-likelihood (-2 * log-likelihood).

    Parameters
    ----------
    data : torch.Tensor
        Observation matrix of shape (n, T), may contain NaNs.
    Fk : torch.Tensor
        Basis function matrix of shape (n, K).
    M : torch.Tensor
        Symmetric matrix of shape (K, K).
    s : float
        Scalar multiplier.
    Depsilon : torch.Tensor
        Diagonal matrix of shape (n, n).
    dtype : torch.dtype, optional
        Data type for computations (default: torch.float64).
    device : str or torch.device, optional
        Device for computations ('cpu' or 'cuda', default: 'cpu').

    Returns
    -------
    float
        Negative log-likelihood value.
    """
    non_missing_points_matrix = ~torch.isnan(data)
    num_columns = data.shape[1]

    n2loglik = non_missing_points_matrix.sum() * torch.log(torch.tensor(2 * torch.pi, dtype=dtype, device=device))
    R = s * Depsilon
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    K = Fk.shape[1]
    L = Fk @ eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0.0))) @ eigenvectors.T
    
    for t in range(num_columns):
        mask = non_missing_points_matrix[:, t]
        zt = data[mask, t]

        # skip all-missing column
        if zt.numel() == 0:
            continue

        Rt = R[mask][:, mask]
        Lt = L[mask]

        log_det = calculateLogDeterminant(R     = Rt, 
                                          L     = Lt, 
                                          K     = K, 
                                          dtype = dtype,
                                          device= device
                                          )
        inv_cz_val = invCz(R        = Rt, 
                           L        = Lt, 
                           z        = zt,
                           dtype    = dtype,
                           device   = device
                           )
        n2loglik += log_det + torch.sum(zt * inv_cz_val)

    return n2loglik.item()

# using in computeLikelihood
# check = ok
def calculateLogDeterminant(
    R: torch.Tensor,
    L: torch.Tensor,
    K: int=None,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> float:
    """
    Calculate the log-determinant of a matrix for likelihood computations.

    Parameters
    ----------
    R : torch.Tensor
        Positive-definite matrix of shape (p, p).
    L : torch.Tensor
        Matrix of shape (p, K).
    K : int, optional
        Number of columns of L to consider (default: all columns of L).
    dtype : torch.dtype, optional
        Data type for computations (default: torch.float64).
    device : str or torch.device, optional
        Device for computations ('cpu' or 'cuda', default: 'cpu').

    Returns
    -------
    float
        Log-determinant value of the matrix.
    """
    if K is None:
        K = L.shape[1]

    first_part_determinant = torch.logdet(torch.eye(K, dtype=dtype, device=device) + L.T @ torch.linalg.solve(R, L))
    second_part_determinant = torch.logdet(R)

    return (first_part_determinant + second_part_determinant).item()

# using in cMLEsp
# check = ok
def logDeterminant(
    mat: torch.Tensor
) -> torch.Tensor:
    """
    Compute the log-determinant of a square matrix.

    Parameters
    ----------
    mat : torch.Tensor
        Square matrix of shape (n, n). The matrix should be positive-definite
        or symmetric to ensure a real-valued log-determinant.

    Returns
    -------
    torch.Tensor
        Log-determinant of the input matrix.
    """
    return torch.logdet(mat.abs())

# usinig in tps_spherical
# check = ok
def build_integral_table(
    func: Callable[[float], float],
    a: float = 0.0,
    b: float = 1.0,
    num_steps: int = 10000,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Precompute a numerical integration lookup table for a given 1D function.

    This function evaluates the cumulative integral of `func` from `a` to each
    point in [a, b], using numerical quadrature and returns sampled points and
    their corresponding integral values as tensors. The result can be used for
    fast interpolation-based integral lookup in later computations.

    Parameters
    ----------
    func : Callable[[float], float], optional
        Function to integrate. Just like `lambda t: np.log1p(-t) / t`.
    a : float, optional
        Lower limit of the integration interval. Default is 0.0.
    b : float, optional
        Upper limit of the integration interval. Default is 1.0.
    num_steps : int, optional
        Number of discrete points to sample between `a` and `b`. Default is 10000.
    eps : float, optional
        Small value to avoid numerical singularities at the integration bounds.
        Default is 1e-12.
    dtype : torch.dtype, optional
        Data type of the output tensors. Default is `torch.float64`.
    device : Union[torch.device, str], optional
        Device on which to store the tensors. Default is `"cpu"`.

    Returns
    -------
    dict of torch.Tensor
        Dictionary containing:
        - "x" : Sampled x-values as a 1D tensor of shape (num_steps,).
        - "y" : Corresponding integrated values of shape (num_steps,).
    """
    x = np.linspace(a + eps, b - eps, num_steps)
    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        y[i], _ = quad(func, a, xi, epsabs=eps)
    x = torch.tensor(x, dtype=dtype, device=device)
    y = torch.tensor(y, dtype=dtype, device=device)

    return {
        "x": x,
        "y": y
    }

# usinig in tps_spherical
# check = ok
def integral_interpolator(
    upper: torch.Tensor,
    lower: torch.Tensor,
    integral_table: dict,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Compute the definite integral between `lower` and `upper` bounds using a precomputed integral table.

    The function interpolates integral values from a lookup table generated by
    :func:`integral_table` to efficiently approximate the integral of a function
    between two tensor-valued limits.

    Parameters
    ----------
    upper : torch.Tensor
        Tensor of upper integration limits. Must lie within the range of the
        table's x-values.
    lower : torch.Tensor
        Tensor of lower integration limits. Must have the same shape as `upper`.
    integral_table : dict
        Dictionary containing precomputed tensors "x" and "y" as returned
        by :func:`integral_table`.
    eps : float, optional
        Small numerical offset to prevent boundary overflow in interpolation.
        Default is 1e-12.

    Returns
    -------
    torch.Tensor
        Tensor of definite integral values, corresponding to the area between
        each pair of lower and upper limits.
    """
    x = integral_table['x']
    y = integral_table['y']
    num_steps = x.numel()

    def interp(
        val: torch.Tensor
    ) -> torch.Tensor:
        idx_float = torch.clamp((val - x[0]) / (x[-1] - x[0]) * (num_steps - 1), 0, num_steps - 2 - eps)
        y_lower = y[idx_float.floor().long()]
        y_upper = y[idx_float.floor().long() + 1]
        t = idx_float - idx_float.floor()
        return y_lower + t * (y_upper - y_lower)

    integral_upper = interp(upper)
    integral_lower = interp(lower)
    return integral_upper - integral_lower

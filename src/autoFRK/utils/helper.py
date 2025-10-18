"""
Title: Setup file of autoFRK-Python Project
Author: Hsu, Yao-Chih
Version: 1141018
Reviewer: 
Reviewed Version:
Reference: 
Description: 
"""

# development only
#import os
#import sys
#sys.path.append(os.path.abspath("./src"))

# import modules
import torch
import numpy as np
import faiss
import gc
from typing import Optional, Dict, Union, Any
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import eigsh
from autoFRK.utils.logger import setup_logger

# logger config
LOGGER = setup_logger()

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
    The fast mode for autoFRK by using KNN for missing data imputation.

    Parameters:
        data: (N, T) or (samples, time_points) tensor.
        loc: (N, spatial_dim) tensor, e.g., 2D space N x 2.
        n_neighbor: Number of neighbors to use for KNN.
        
    Returns:
        torch.Tensor: The data tensor with missing values imputed.
    """
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
    
    """
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
    calculate_with_spherical: bool = False,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = 'cpu'
) -> torch.Tensor:
    """
    
    """
    # 去除全為 NaN 的欄位
    not_all_nan = ~torch.isnan(data).all(dim=0)
    data = data[:, not_all_nan]

    # 檢查資料中是否有缺失值
    is_data_with_missing_values = torch.isnan(data).any()

    # 找出整行都是 NaN 的列（完全缺失）
    na_rows = torch.isnan(data).all(dim=1)
    pick = torch.arange(data.shape[0], dtype=torch.int64, device=device)
    if na_rows.any():
        data = data[~na_rows]
        loc = loc[~na_rows]  # 同步刪除 loc 中相同的行  # need fix
        D = D[~na_rows][:, ~na_rows]
        pick = pick[~na_rows]
        is_data_with_missing_values = torch.isnan(data).any()

    # 如果 D 未提供，則初始化為單位對角矩陣
    if D is None:
        D = torch.eye(data.shape[0], dtype=dtype, device=device)

    # 取得位置維度
    d = loc.shape[1]

    # 計算 klim 與選 knot
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

    # 處理 K 值
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

    # Fk 為 None 時初始化 basis function 值
    if Fk is None:
        from autoFRK.mrts import MRTS
        mrts = MRTS(dtype   = dtype,
                    device  = device
                    )
        Fk = mrts.forward(knot                      = knot,
                          k                         = max(K),
                          x                         = loc,
                          maxknot                   = max_knot,
                          calculate_with_spherical  = calculate_with_spherical,
                          dtype                     = dtype,
                          device                    = device
                          )

        # old version
        #from autoFRK.mrts_old import MRTS
        #mrts = MRTS(locs    = loc,
        #            k       = max(K),
        #            dtype   = dtype,
        #            device  = device
        #            )  # 待修 (knot, max(K), loc, max_knot) need fix
        #Fk = mrts.forward()

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
                data = fast_mode_knn_sklearn(data       = data,
                                             loc        = loc, 
                                             n_neighbor = num_neighbors
                                             )
            elif method == "fast_faiss":  # have OpenMP issue
                data = fast_mode_knn_faiss(data         = data,
                                           loc          = loc, 
                                           n_neighbor   = num_neighbors
                                           )
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
) -> dict:
    """
    Compute negative log-likelihood.
    
    Parameters:
        nrow_Fk: (int) Number of rows in basis matrix Fk.
        ncol_Fk: (int) Number of basis functions (columns of Fk).
        s: (int) Effective sample size.
        p: (int) Number of variables (e.g. number of spatial points).
        matrix_JSJ: (torch.Tensor) Covariance-like matrix (should be symmetric).
        sample_covariance_trace: (float) Trace of sample covariance matrix.
        vfixed: (float, optional) Fixed noise variance (if provided).
        ldet: (float, optional) Log determinant of transformation matrix.
        device: (str) 'cpu' or 'cuda'.

    Returns:
        dict: {
            'negative_log_likelihood': float,
            'P': torch.Tensor (eigenvectors),
            'v': float,
            'd_hat': torch.Tensor
        }
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
    Estimate the v parameter.

    Parameters:
        d: (torch.Tensor) 1D tensor of nonnegative eigenvalues (length k)
        s: (float) A positive numeric constant
        sample_covariance_trace: (float) Trace of sample covariance
        n: (int) Sample size

    Returns:
        v: (float) Estimated noise variance
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

    Parameters:
        d: (torch.Tensor) 1D tensor of nonnegative values (eigenvalues)
        s: (float) A positive numeric
        v: (float) A positive numeric

    Returns:
        torch.Tensor: A tensor of estimated eta values
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
    Estimate the negative log-likelihood (up to constant)

    Parameters:
        d: Tensor of nonnegative values (eigenvalues)
        s: A positive scalar
        v: A positive scalar
        sample_covariance_trace: Scalar trace value
        sample_size: Number of samples (int)

    Returns:
        Scalar negative log-likelihood value
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
    Compute negative log-likelihood (-2 * log(likelihood)).

    Parameters:
        data (n x T): Observation matrix with possible NaNs.
        Fk (n x K): Basis function matrix.
        M (K x K): Symmetric matrix.
        s (float): Scalar multiplier.
        Depsilon (n x n): Diagonal matrix.
        device: CPU or GPU.

    Returns:
        float: Negative log-likelihood value.
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
    Internal function: calculate the log determinant for the likelihood use.

    Parameters:
        R (torch.Tensor): (p x p) positive-definite matrix
        L (torch.Tensor): (p x K) matrix
        K (int): A numeric
        device (str or torch.device): computation device

    Returns:
        float: log-determinant value
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
    Internal function: log-determinant of a square matrix

    Parameters:
        mat (torch.Tensor): 
            Square matrix whose log-determinant will be computed.

    Returns:
        torch.Tensor:
            The log-determinant of the input matrix.
    """
    return torch.logdet(mat.abs())

"""
Title: Setup file of autoFRK-Python Project
Author: Hsu, Yao-Chih
Version: 1141012
Reference:
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
from typing import Optional, Union, Any
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize_scalar
from autoFRK.utils.logger import setup_logger
from autoFRK.mrts import MRTS

# logger config
LOGGER = setup_logger()

# change into tensor, using in autoFRK
# check = ok
def to_tensor(
    obj: Any,
    dtype = torch.float64,
    device: Union[torch.device, str] = 'cpu'
) -> torch.Tensor:
    """
    Recursively convert any input object to torch.Tensor.
    Handles: numbers, lists/tuples, np.ndarray, nested dicts.

    Args:
        obj: int, float, list, tuple, np.ndarray, dict, torch.Tensor
        dtype: desired torch dtype
        device: target device

    Returns:
        torch.Tensor or dict (nested with tensors)
    """
    if isinstance(obj, torch.Tensor) and ():
        if obj.dtype != dtype or obj.device != device:
            t = obj.to(dtype=dtype, device=device)
        else:
            t = obj
    elif isinstance(obj, (int, float)):
        t = torch.tensor(obj, dtype=dtype, device=device)
    elif isinstance(obj, (list, tuple)):
        converted = [to_tensor(x, dtype=dtype, device=device) for x in obj]
        try:
            t = torch.stack(converted)
        except:
            t = converted
    elif isinstance(obj, dict):
        t = {k: to_tensor(v, dtype=dtype, device=device) for k, v in obj.items()}
    elif hasattr(obj, 'shape'):
        t = torch.tensor(obj, dtype=dtype, device=device)
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")
    
    return t

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
    Fk: torch.Tensor = None,
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
    pick = torch.arange(data.shape[0], dtype=dtype, device=device)
    if na_rows.any():
        data = data[~na_rows]
        loc = loc[~na_rows]  # 同步刪除 loc 中相同的行 need fix
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
        mrts = MRTS(locs    = loc,
                    k       = max(K),
                    dtype   = dtype,
                    device  = device
                    )  # 待修 (knot, max(K), loc, max_knot) need fix
        Fk = mrts.forward()

    AIC_list = [float('inf')] * len(K)
    num_data_columns = data.shape[1]

    if method == "EM" and DfromLK is None:
        for k in range(len(K)):
            AIC_list[k] = indeMLE(data  = data,
                                  Fk    = Fk[pick, :K[k]],
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
            data = fast_mode_knn_sklearn(data=data, loc=loc, n_neighbor=num_neighbors) 
        if DfromLK is None:
            iD = torch.linalg.solve(D, torch.eye(D.shape[0], dtype=dtype, device=device))
            iDFk = iD @ Fk[pick, :]
            iDZ = iD @ data
        else:
            wX = DfromLK["wX"][pick, :]
            G = DfromLK["wX"].T @ DfromLK["wX"] + DfromLK["lambda"] * DfromLK["Q"]
            weight = DfromLK["weights"][pick]
            wwX = torch.diag(torch.sqrt(weight)) @ wX
            wXiG = torch.linalg.solve(G, wwX.T).T
            iDFk = weight * Fk[pick, :] - wXiG @ (wwX.T @ Fk[pick, :])
            iDZ = weight * data - wXiG @ (wwX.T @ data)

        sample_covariance_trace = torch.sum(iDZ * data) / num_data_columns

        for k in range(len(K)):
            Fk_k = Fk[pick, :K[k]]
            iDFk_k = iDFk[:, :K[k]]
            inverse_square_root_matrix = get_inverse_square_root_matrix(left_matrix  = Fk_k,
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
    out = Fk[:, :Kopt]
    return out

# check = ok
def get_inverse_square_root_matrix(
    left_matrix,
    right_matrix
):
    """
    
    """
    mat = left_matrix.T @ right_matrix  # A^T * B
    mat = (mat + mat.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(mat)
    inv_sqrt_eigvals = torch.diag(torch.clamp(eigvals, min=1e-10).rsqrt())
    return eigvecs @ inv_sqrt_eigvals @ eigvecs.T

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
            kconst *= nmbin[kk]

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

# compute negative log likelihood for autoFRK, using in selectBasis
# check = ok
def cMLE(
    Fk: torch.Tensor,
    num_columns: int,
    sample_covariance_trace: float,
    inverse_square_root_matrix: torch.Tensor,
    matrix_JSJ: torch.Tensor,
    s: float = 0,
    ldet: float = 0,
    wSave: bool = False,
    onlylogLike: bool = None,
    vfixed: float = None,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = 'cpu'
) -> dict:
    """
    Internal function: maximum likelihood estimate with the likelihood

    Parameters:
        Fk: (N, K) torch.Tensor, basis functions.
        num_columns: (int) Number of columns in the data.
        sample_covariance_trace: (float) Trace of the sample covariance matrix.
        inverse_square_root_matrix: (N, K) torch.Tensor, inverse square root matrix.
        matrix_JSJ: (K, K) torch.Tensor, covariance-like matrix.
        s: (float) Effective sample size, default is 0.
        ldet: (float) Log determinant of the transformation matrix, default is 0.
        wSave: (bool) Whether to save the L matrix, default is False.
        onlylogLike: (bool) If True, only return the negative log likelihood.
        vfixed: (float, optional) Fixed noise variance, if provided.
        device: (str) 'cpu' or 'cuda'.

    Returns:
        dict: {
            'v': (float) Estimated noise variance,
            'M': (torch.Tensor) Matrix M,
            's': (int) Effective sample size,
            'negloglik': (float) Negative log likelihood,
            'L': (torch.Tensor, optional) L matrix if wSave is True.
        }
    """
    nrow_Fk = Fk.shape[0]

    likelihood_object = computeNegativeLikelihood(
        nrow_Fk                 = nrow_Fk,
        ncol_Fk                 = Fk.shape[1],
        s                       = s,
        p                       = num_columns,
        matrix_JSJ              = matrix_JSJ,
        sample_covariance_trace = sample_covariance_trace,
        vfixed                  = vfixed,
        ldet                    = ldet,
        dtype                   = dtype,
        device                  = device
    )

    negative_log_likelihood = likelihood_object['negative_log_likelihood']

    if onlylogLike:
        return {'negloglik': negative_log_likelihood}

    P = likelihood_object['P']
    d_hat = likelihood_object['d_hat']
    v = likelihood_object['v']
    M = inverse_square_root_matrix @ P @ (torch.diag(d_hat) @ P.T) @ inverse_square_root_matrix

    if not wSave:
        L = None
    elif d_hat[0] != 0:
        L = Fk @ ((torch.diag(torch.sqrt(d_hat)) @ P.T) @ inverse_square_root_matrix)
        L = L[:, d_hat > 0]
    else:
        L = torch.zeros((nrow_Fk, 1), dtype=dtype, device=device)

    return {'v': v,
            'M': M,
            's': s,
            'negloglik': negative_log_likelihood,
            'L': L
            }

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

# independent maximum likelihood estimation for autoFRK, using in selectBasis
# check = ok
def indeMLE(
    data: torch.Tensor,
    Fk: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    maxit: int = 50,
    avgtol: float = 1e-6,
    wSave: bool = False,
    DfromLK: Optional[dict] = None,
    vfixed: Optional[float] = None,
    verbose: bool = True,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> dict:
    """
    Internal function: indeMLE

    Parameters:
        data (torch.Tensor): 
            n x T data matrix (NA allowed), where each column is z[t].
        Fk (torch.Tensor): 
            n x K matrix of basis function values; each column is a basis function evaluated at observation locations.
        D (torch.Tensor, optional): 
            n x n diagonal matrix.
        maxit (int, default 50): 
            Maximum number of iterations.
        avgtol (float, default 1e-6): 
            Average tolerance for convergence.
        wSave (bool, default False): 
            Whether to compute and return weight and covariance matrices.
        DfromLK (dict, optional): 
            n x n matrix or dictionary of low-rank kernel precomputations.
        vfixed (float, optional): 
            Fixed variance parameter (if provided, overrides estimation).
        verbose (bool, default True): 
            Print useful information during computation.
        device (str or torch.device, default 'cpu'): 
            Device to perform computation on.

    Returns:
        dict
            Dictionary containing estimated matrices, variance parameters, and optional diagnostic information.
    """
    withNA = torch.isnan(data).any().item()

    TT = data.shape[1]
    empty = torch.isnan(data).all(dim=0)
    notempty = (~empty).nonzero(as_tuple=True)[0]
    if empty.any():
        data = data[:, notempty]

    del_rows = torch.isnan(data).all(dim=1).nonzero(as_tuple=True)[0]
    pick = torch.arange(data.shape[0], dtype=torch.int64, device=device)

    if D is None:
        D = torch.eye(data.shape[0], dtype=dtype, device=device)

    if not isDiagonal(D):
        D0 = D
    else:
        D0 = torch.diag(torch.diag(D))

    if withNA and len(del_rows) > 0:
        pick = pick[~torch.isin(pick, del_rows)]
        data = data[~torch.isin(torch.arange(data.shape[0], dtype=torch.int64, device=device), del_rows), :]
        Fk = Fk[~torch.isin(torch.arange(Fk.shape[0], dtype=dtype, device=device), del_rows), :]
        if not torch.allclose(D, torch.diag(torch.diagonal(D))):
            D = D[~torch.isin(torch.arange(D.shape[0], dtype=dtype, device=device), del_rows)][:, ~torch.isin(torch.arange(D.shape[1], dtype=dtype, device=device), del_rows)]
        else:
            keep_mask = ~torch.isin(torch.arange(D.shape[0], dtype=torch.int64, device=device), del_rows)
            full_diag = torch.zeros(D.shape[0], dtype=dtype, device=device)
            full_diag[keep_mask] = torch.diagonal(D)[keep_mask]
            D = torch.diag(full_diag)
        withNA = torch.isnan(data).any().item()

    N = data.shape[0]
    K = Fk.shape[1]
    Depsilon = D
    is_diag = torch.allclose(D, torch.diag(torch.diagonal(D)))
    mean_diag = torch.mean(torch.diagonal(D))
    isimat = is_diag and torch.allclose(torch.diagonal(Depsilon), mean_diag.repeat(N), atol=1e-10)

    if not withNA:
        if isimat and DfromLK is None:
            sigma = 0  # we cannot find `.Option$sigma_FRK` in the R code  # outside
            out = cMLEimat(Fk           = Fk, 
                           data         = data, 
                           s            = sigma, 
                           wSave        = wSave,
                           S            = None,
                           onlylogLike  = None,
                           dtype        = dtype,
                           device       = device
                           )
            if out['v'] is not None:
                out['s'] = out['v'] if sigma == 0 else sigma
                out.pop("v", None)
            if wSave:
                w = torch.zeros((K, TT), dtype=dtype, device=device)
                w[:, notempty] = out['w']
                out['w'] = w
                out['pinfo'] = {'D': D0, 
                                'pick': pick
                                }
            return out
        
        elif DfromLK is None:
            out = cMLEsp(Fk         = Fk, 
                         data       = data, 
                         Depsilon   = Depsilon, 
                         wSave      = wSave,
                         dtype      = dtype,
                         device     = device
                         )
            if wSave:
                w = torch.zeros((K, TT), dtype=dtype, device=device)
                w[:, notempty] = out['w']
                out['w'] = w
                out['pinfo'] = {'D': D0, 
                                'pick': pick
                                }
            return out
        
        else:
            out = cMLElk(Fk         = Fk, 
                         data       = data, 
                         Depsilon   = Depsilon, 
                         wSave      = wSave, 
                         DfromLK    = DfromLK, 
                         vfixed     = vfixed,
                         dtype      = dtype,
                         device     = device
                         )
            if wSave:
                w = torch.zeros((K, TT), dtype=dtype, device=device)
                w[:, notempty] = out['w']
                out['w'] = w
            return out
        
    else:
        out = EM0miss(Fk        = Fk, 
                      data      = data, 
                      Depsilon  = Depsilon, 
                      maxit     = maxit, 
                      avgtol    = avgtol, 
                      wSave     = wSave,
                      DfromLK   = DfromLK, 
                      vfixed    = vfixed, 
                      verbose   = verbose,
                      dtype     = dtype,
                      device    = device
                      )
        if wSave:
            w = torch.zeros((K, TT), dtype=dtype, device=device)
            w[:, notempty] = out['w']
            out['w'] = w
            if DfromLK is None:
                out['pinfo'] = {'D': D0,
                                'pick': pick
                                }
        return out

# convert dense tensor to sparse matrix, using in indeMLE
# python 不需要，在 R 中僅作為節省記憶體的角色
# def toSparseMatrix(
#     mat: torch.Tensor, 
#     verbose: bool=False
# ) -> torch.Tensor:
#     """
    
#     """
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

# using in indeMLE
# check = ok
def cMLEimat(
    Fk: torch.Tensor,
    data: torch.Tensor,
    s: float,
    wSave: bool = False,
    S: Optional[torch.Tensor] = None,
    onlylogLike: Optional[bool] = None,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> dict:
    """
    
    """
    if onlylogLike is None:
        onlylogLike = not wSave

    num_columns = data.shape[1]
    nrow_Fk, ncol_Fk = Fk.shape

    projection = computeProjectionMatrix(Fk1    = Fk, 
                                         Fk2    = Fk, 
                                         data   = data, 
                                         S      = S, 
                                         dtype  = dtype, 
                                         device = device
                                         )
    inverse_square_root_matrix = projection["inverse_square_root_matrix"]
    matrix_JSJ = projection["matrix_JSJ"]

    sample_covariance_trace = torch.sum(data ** 2) / num_columns

    likelihood_object = computeNegativeLikelihood(nrow_Fk                   = nrow_Fk,
                                                  ncol_Fk                   = ncol_Fk,
                                                  s                         = s,
                                                  p                         = num_columns,
                                                  matrix_JSJ                = matrix_JSJ,
                                                  sample_covariance_trace   = sample_covariance_trace,
                                                  vfixed                    = None,
                                                  ldet                      = 0.0,
                                                  dtype                     = dtype,
                                                  device                    = device
                                                  )

    negative_log_likelihood = likelihood_object["negative_log_likelihood"]

    if onlylogLike:
        return {"negloglik": negative_log_likelihood}

    P = likelihood_object["P"]
    d_hat = likelihood_object["d_hat"]
    v = likelihood_object["v"]

    M = inverse_square_root_matrix @ P @ (P.T * d_hat[:, None]) @ inverse_square_root_matrix

    if not wSave:
        return {"v": v, 
                "M": M, 
                "s": s, 
                "negloglik": negative_log_likelihood
                }

    L = Fk @ ((torch.diag(torch.sqrt(d_hat)) @ P.T) @ inverse_square_root_matrix).T

    if ncol_Fk > 2:
        reduced_columns = torch.cat([
            torch.tensor([0], dtype=torch.int64, device=device),
            (d_hat[1:(ncol_Fk - 1)] > 0).nonzero(as_tuple=True)[0]
        ])
    else:
        reduced_columns = torch.tensor([ncol_Fk - 1], dtype=torch.int64, device=device)

    L = L[:, reduced_columns]

    invD = torch.ones(nrow_Fk, dtype=dtype, device=device) / (s + v)
    iDZ = invD[:, None] * data

    right = L @ (torch.linalg.inv(torch.eye(L.shape[1], dtype=dtype, device=device) + L.T @ (invD[:, None] * L)) @ (L.T @ iDZ))

    INVtZ = iDZ - invD[:, None] * right
    etatt = M @ Fk.T @ INVtZ

    GM = Fk @ M

    diag_matrix = (s + v) * torch.eye(nrow_Fk, dtype=dtype, device=device)

    V = M - GM.T @ invCz(R      = diag_matrix,
                         L      = L, 
                         z      = GM,
                         dtype  = dtype,
                         device = device
                         ).T

    return {"v": v,
            "M": M,
            "s": s,
            "negloglik": negative_log_likelihood,
            "w": etatt,
            "V": V
            }

# using in cMLEimat
# check = ok
def computeProjectionMatrix(
    Fk1: torch.Tensor, 
    Fk2: torch.Tensor, 
    data: torch.Tensor, 
    S: torch.Tensor=None,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> dict:
    """
    Internal function: maximum likelihood estimate with the likelihood

    Parameters:
        Fk1 (torch.Tensor): (n, K) matrix
        Fk2 (torch.Tensor): (n, K) matrix
        data (torch.Tensor): (n, T) matrix
        S (torch.Tensor or None): (n, n) matrix
        device (str): "cpu" or "cuda"
 
    Returns:
        dict: {
            'inverse_square_root_matrix': torch.Tensor,
            'matrix_JSJ': torch.Tensor
        }
    """
    if S is not None:
        S = to_tensor(S, dtype=dtype, device=device)

    num_columns = data.shape[1]
    inverse_square_root_matrix = getInverseSquareRootMatrix(A = Fk1, 
                                                            B = Fk2
                                                            )
    inverse_square_root_on_Fk2 = inverse_square_root_matrix @ Fk2.T

    if S is None:
        matrix_JSJ = (inverse_square_root_on_Fk2 @ data) @ (inverse_square_root_on_Fk2 @ data).T / num_columns
    else:
        matrix_JSJ = (inverse_square_root_on_Fk2 @ S) @ inverse_square_root_on_Fk2.T

    matrix_JSJ = (matrix_JSJ + matrix_JSJ.T) / 2

    return {"inverse_square_root_matrix": inverse_square_root_matrix,
            "matrix_JSJ": matrix_JSJ
            }

# using in computeProjectionMatrix
# check = ok
def getInverseSquareRootMatrix(
    A: torch.Tensor, 
    B: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute inverse square root matrix of (A.T @ B), assuming it is symmetric.
    
    Parameters:
        A: Tensor of shape (n, k)
        B: Tensor of shape (n, k)
        device: 'cpu' or 'cuda'
        
    Returns:
        Inverse square root of (A.T @ B): Tensor of shape (k, k)
    """
    mat = A.T @ B

    eigenvalues, eigenvectors = torch.linalg.eigh(mat)
    eigvals_clamped = torch.clamp(eigenvalues, min=eps)
    inv_sqrt_eigvals = torch.diag(eigvals_clamped.rsqrt())

    return eigenvectors @ inv_sqrt_eigvals @ eigenvectors.T

# using in cMLEimat
# check = ok
def invCz(
    R: torch.Tensor, 
    L: torch.Tensor, 
    z: torch.Tensor, 
    dtype: torch.dtype=torch.float64,
    device: Union[torch.device, str]='cpu'
) -> torch.Tensor:
    """


    Parameters:
        R: (p x p) positive definite matrix
        L: (p x K) matrix
        z: (p,) vector or (1 x p) row matrix
        device: 'cpu' or 'cuda', or torch.device

    Returns:
        (1 x p) tensor
    """
    if z.dim() == 1:
        z = z.unsqueeze(1)

    K = L.shape[1]
    iR = torch.linalg.pinv(R)
    iRZ = iR @ z
    right = L @ torch.linalg.inv(torch.eye(K, dtype=dtype ,device=device) + (L.T @ iR @ L)) @ (L.T @ iRZ) 
    result = iRZ - iR @ right

    return result.T

# using in indeMLE
# check = ok, but have some problem
def EM0miss(
    Fk: torch.Tensor, 
    data: torch.Tensor, 
    Depsilon: torch.Tensor, 
    maxit: int=100, 
    avgtol: float=1e-4, 
    wSave: bool=False, 
    DfromLK: dict=None,
    vfixed: float=None,
    verbose: bool=True,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> dict:
    """

    """
    O = ~torch.isnan(data)
    TT = data.shape[1]
    ncol_Fk = Fk.shape[1]

    ziDz = torch.full((TT,), float('nan'), device=device)
    ziDB = torch.full((TT, ncol_Fk), float('nan'), device=device)
    db = {}
    D = Depsilon
    iD = torch.linalg.inv(D)
    diagD = isDiagonal(D)

    if DfromLK is not None:
        DfromLK = to_tensor(DfromLK, dtype=dtype, device=device)
        pick = DfromLK.get("pick", None)
        weights = DfromLK["weights"]
        if pick is None:
            pick = torch.arange(len(weights), dtype=torch.int64, device=device)
        else:
            pick = to_tensor(pick, dtype=torch.int64, device=device)
        weight = weights[pick]
        DfromLK["wX"] = DfromLK["wX"][pick, :]
        wwX = torch.diag(torch.sqrt(weight)) @ DfromLK["wX"]
        lQ = DfromLK["lambda"] * DfromLK["Q"]

    for tt in range(TT):
        if DfromLK is not None:
            obs_idx = O[:, tt].bool()
            iDt = None
            if obs_idx.sum() == O.shape[0]:
                wXiG = wwX @ torch.linalg.inv(DfromLK["G"])
            else:
                wX_obs = DfromLK["wX"][obs_idx, :]
                G = wX_obs.T @ wX_obs + lQ
                wXiG = wwX[obs_idx, :] @ torch.linalg.inv(G)

            Bt = Fk[obs_idx, :]
            if Bt.ndim == 1:
                Bt = Bt.unsqueeze(0)

            iDBt = weight[obs_idx].unsqueeze(1) * Bt - wXiG @ (wwX[obs_idx, :].T @ Bt)
            zt = data[obs_idx, tt]
            ziDz[tt] = torch.sum(zt * (weight[obs_idx] * zt - wXiG @ (wwX[obs_idx, :].T @ zt)))
            ziDB[tt, :] = (zt @ iDBt).squeeze()
            BiDBt = Bt.T @ iDBt

        else:
            if not diagD:
                iDt = torch.linalg.inv(D[obs_idx][:, obs_idx])
            else:
                iDt = iD[obs_idx][:, obs_idx]

            Bt = Fk[obs_idx, :]
            if Bt.ndim == 1:
                Bt = Bt.unsqueeze(0)

            iDBt = iDt @ Bt
            zt = data[obs_idx, tt]
            ziDz[tt] = torch.sum(zt * (iDt @ zt))
            ziDB[tt, :] = (zt @ iDBt).squeeze()
            BiDBt = Bt.T @ iDBt

        db[tt] = {"iDBt": iDBt,
                  "zt": zt,
                  "BiDBt": BiDBt
                  }

    del iDt, Bt, iDBt, zt, BiDBt
    _ = gc.collect()
    torch.cuda.empty_cache()

    dif = float("inf")
    cnt = 0
    Z0 = data.clone()
    Z0[torch.isnan(Z0)] = 0
    old = cMLEimat(Fk           = Fk, 
                   data         = Z0, 
                   s            = 0, 
                   wSave        = True,
                   S            =  None,
                   onlylogLike  =  None,
                   dtype        = dtype,
                   device       = device
                   )
    if vfixed is None:
        old["s"] = old["v"]
    else:
        old["s"] = to_tensor(vfixed)
    old["M"] = convertToPositiveDefinite(mat    = old["M"],
                                         dtype  = dtype,
                                         device = device
                                         )
    Ptt1 = old["M"]

    while (dif > (avgtol * (100 * (ncol_Fk ** 2)))) and (cnt < maxit):
        etatt = torch.zeros((ncol_Fk, TT), dtype=dtype, device=device)
        sumPtt = torch.zeros((ncol_Fk, ncol_Fk), dtype=dtype, device=device)
        s1 = torch.zeros(TT, dtype=dtype, device=device)

        for tt in range(TT):            
            ginv_Ptt1 = torch.linalg.pinv(convertToPositiveDefinite(mat     = Ptt1,
                                                                    dtype   = dtype,
                                                                    device  = device
                                                                    ))
            iP = convertToPositiveDefinite(mat      = ginv_Ptt1 + BiDBt / old["s"],
                                           dtype    = dtype,
                                           device   = device
                                           )
            Ptt = torch.linalg.inv(iP)  # will broken under some situation  # need fix
            Gt = (Ptt @ iDBt.T) / old["s"]
            eta = Gt @ zt
            s1kk = torch.diagonal(BiDBt @ (eta.unsqueeze(1) @ eta.unsqueeze(0) + Ptt))
            
            sumPtt += Ptt
            etatt[:, tt] = eta
            s1[tt] = torch.sum(s1kk)

        if vfixed is None:
            s = torch.max(
                (torch.sum(ziDz) - 2 * torch.sum(ziDB * etatt.T) + torch.sum(s1)) / torch.sum(O),
                torch.tensor(1e-8, dtype=dtype, device=device)
            )
            new = {"M": (etatt @ etatt.T + sumPtt) / TT,
                   "s": s,
                   }
        else:
            new = {"M": (etatt @ etatt.T + sumPtt) / TT,
                   "s": vfixed,
                   }

        new["M"] = (new["M"] + new["M"].T) / 2
        dif = torch.sum(torch.abs(new["M"] - old["M"])) + torch.abs(new["s"] - old["s"])
        cnt += 1
        old = new
        Ptt1 = old["M"]

    if verbose:
        info_msg = f'Number of iteration: {cnt}'
        LOGGER.info(info_msg)
        
    n2loglik = computeLikelihood(data       = data,
                                 Fk         = Fk,
                                 M          = new["M"],
                                 s          = new["s"],
                                 Depsilon   = Depsilon,
                                 dtype      = dtype,
                                 device     = device
                                 )

    if not wSave:
        return {
            "M": new["M"],
            "s": new["s"],
            "negloglik": n2loglik
        }

    elif DfromLK is not None:
        out = {
            "M": new["M"],
            "s": new["s"],
            "negloglik": n2loglik,
            "w": etatt,
            "V": new["M"] - (etatt @ etatt.T) / TT
        }

        eigenvalues, eigenvectors = torch.linalg.eigh(new["M"])
        L = Fk @ eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0.0)))

        weight = DfromLK["weights"][pick]
        wlk = torch.full((lQ.shape[0], TT), float("nan"), device=device)

        for tt in range(TT):
            obs_idx = O[:, tt].bool()
            if torch.sum(obs_idx) == O.shape[0]:
                wXiG = wwX @ torch.linalg.solve(DfromLK["G"], torch.eye(DfromLK["G"].shape[0], dtype=dtype, device=device))
            else:
                wX_tt = DfromLK["wX"][obs_idx]
                G = wX_tt.T @ wX_tt + lQ
                wXiG = wwX[obs_idx] @ torch.linalg.solve(G, torch.eye(G.shape[0], dtype=dtype, device=device))

            dat = data[obs_idx, tt]
            Lt = L[obs_idx]
            iDL = weight[obs_idx].unsqueeze(1) * Lt - wXiG @ (wwX[obs_idx].T @ Lt)
            itmp = torch.linalg.solve(
                torch.eye(L.shape[1], dtype=dtype, device=device) + (Lt.T @ iDL) / out["s"],
                torch.eye(L.shape[1], dtype=dtype, device=device)
            )
            iiLiD = itmp @ (iDL.T / out["s"])
            wlk[:, tt] = (wXiG.T @ dat - wXiG.T @ Lt @ (iiLiD @ dat)).squeeze()

        out["pinfo"] = {
            "wlk": wlk, 
            "pick": pick
        }
        out["missing"] = {
            "miss": 1 - O, 
            "maxit": maxit, 
            "avgtol": avgtol
        }
        return out

    else:
        out = {
            "M": new["M"],
            "s": new["s"],
            "negloglik": n2loglik,
            "w": etatt,
            "V": new["M"] - (etatt @ etatt.T) / TT
        }
        out["missing"] = {
            "miss": 1 - O, 
            "maxit": maxit, 
            "avgtol": avgtol
        }
        return out

# using in EM0miss
# check = ok
def isDiagonal(
    tensor: torch.Tensor,
    tol=1e-10
) -> bool:
    """
    Internal function: check if a numeric-like object is diagonal

    Parameters:
        tensor:
        tol:
    
    Return:
        bool: 
    """
    if tensor.numel() == 1:
        return True

    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        return False

    diag = torch.diag(torch.diagonal(tensor))
    return torch.allclose(tensor, diag, atol=tol)

# using in EM0miss
# check = ok
def convertToPositiveDefinite(
    mat: torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> torch.Tensor:
    """
    Internal function: convert a matrix to positive definite

    Parameters:
        mat (torch.Tensor): Input 2D matrix (square, symmetric or not).
        device (torch.device): Device to perform computation on ("cpu" or "cuda").

    Returns:
        torch.Tensor: A positive-definite version of the input matrix.
    """
    # Ensure symmetry
    if not torch.allclose(mat, mat.T, atol=1e-10):
        mat = (mat + mat.T) / 2

    try:
        # Compute eigenvalues only
        eigenvalues = torch.linalg.eigvalsh(mat)
        min_eigenvalue = torch.min(eigenvalues).item()
    except RuntimeError:
        # Fallback in case of numerical error
        mat = (mat + mat.T) / 2
        eigenvalues = torch.linalg.eigvalsh(mat)
        min_eigenvalue = torch.min(eigenvalues).item()

    if min_eigenvalue <= 0:
        adjustment = abs(min_eigenvalue) + 1e-6
        mat = mat + torch.eye(mat.shape[0], dtype=dtype, device=device) * adjustment

    return mat

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

# using in indeMLE
# check = ok
def cMLEsp(
    Fk: torch.Tensor,
    data: torch.Tensor,
    Depsilon: torch.Tensor,
    wSave: bool = False,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> dict:
    """
    Internal function: cMLEsp

    Parameters:
        Fk (torch.Tensor): 
            (n × K) matrix of basis function values at observation locations.

        data (torch.Tensor): 
            (n × T) data matrix (can contain NaN).

        Depsilon (torch.Tensor): 
            (n × n) diagonal covariance matrix (measurement error variances).

        wSave (bool, default=False): 
            Whether to compute and return weight and covariance matrices.

        device (torch.device or str, optional): 
            Device for computation.

        dtype (torch.dtype, optional): 
            Data precision.

    Returns:
        dict
            Dictionary containing:
            - 'M', 's', and (if wSave=True) 'w', 'V'
    """
    iD = torch.linalg.inv(Depsilon)
    ldetD = logDeterminant(mat = Depsilon).item()
    iDFk = iD @ Fk
    num_columns = data.shape[1]

    projection = computeProjectionMatrix(Fk1    = Fk,
                                         Fk2    = iDFk,
                                         data   = data,
                                         S      = None,
                                         dtype  = dtype,
                                         device = device
                                         )
    inverse_square_root_matrix = projection["inverse_square_root_matrix"]
    matrix_JSJ = projection["matrix_JSJ"]

    trS = torch.sum((iD @ data) * data) / num_columns
    out = cMLE(Fk                           = Fk,
               num_columns                  = num_columns,
               sample_covariance_trace      = trS,
               inverse_square_root_matrix   = inverse_square_root_matrix,
               matrix_JSJ                   = matrix_JSJ,
               s                            = 0,
               ldet                         = ldetD,
               wSave                        = wSave,
               onlylogLike                  = None,
               vfixed                       = None,
               dtype                        = dtype,
               device                       = device
               )

    if wSave:
        L = out["L"]
        s_plus_v = out["s"] + out["v"]
        invD = iD / s_plus_v
        iDZ = invD @ data
        right0 = L @ torch.linalg.solve(
            torch.eye(L.shape[1], dtype=dtype, device=device) + L.T @ (invD @ L),
            torch.eye(L.shape[1], dtype=dtype, device=device)
        )

        INVtZ = iDZ - invD @ right0 @ (L.T @ iDZ)
        etatt = out["M"] @ Fk.T @ INVtZ
        out["w"] = etatt
        GM = Fk @ out["M"]
        iDGM = invD @ GM
        out["V"] = out["M"] - GM.T @ (iDGM - invD @ right0 @ (L.T @ iDGM))

    out["s"] = out["v"]
    out.pop("v", None)
    out.pop("L", None)
    return out

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

# using in indeMLE
# check = ok
def cMLElk(
    Fk: torch.Tensor,
    data: torch.Tensor,
    Depsilon: torch.Tensor,
    wSave: bool = False,
    DfromLK: dict = None,
    vfixed: float = None,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> dict:
    """
    Internal function: cMLElk

    Parameters:
        Fk (torch.Tensor):
            (n × K) matrix of basis function values, where each column
            represents a basis function evaluated at the observation locations.

        data (torch.Tensor):
            (n × T) data matrix (can contain NaN) with z[t] as the t-th column.

        Depsilon (torch.Tensor):
            (n × n) diagonal covariance matrix of measurement errors.

        wSave (bool, default=False):
            Whether to compute and return additional weight and covariance matrices.

        DfromLK (dict):
            Dictionary containing precomputed quantities from the low-rank kernel step:
                - 'lambda' (float): regularization parameter.
                - 'pick' (list[int]): indices of selected observations.
                - 'wX' (torch.Tensor): weighted design matrix.
                - 'weights' (torch.Tensor): vector of weights.
                - 'Q' (torch.Tensor): penalty matrix.

        vfixed (float, optional):
            Fixed variance parameter (if provided, overrides estimation).

    Returns:
        dict:
            Dictionary containing:
                - 'M' (torch.Tensor): estimated model matrix.
                - 's' (float): variance parameter.
                - 'w' (torch.Tensor, optional): estimated weights if wSave=True.
                - 'V' (torch.Tensor, optional): covariance matrix of weights.
                - 'pinfo' (dict, optional): diagnostic info with 'wlk' and 'pick'.
    """
    num_columns = data.shape[1]
    lambda_ = DfromLK["lambda"]
    pick = DfromLK["pick"]
    wX = DfromLK["wX"]
    weight = DfromLK["weights"]
    Q = DfromLK["Q"]

    if len(pick) < wX.shape[0]:
        wX = wX[pick, :]
        weight = weight[pick]

    G = wX.T @ wX + lambda_ * Q
    wwX = torch.diag(torch.sqrt(weight)) @ wX
    wXiG = wwX @ torch.linalg.solve(G, torch.eye(G.shape[0], dtype=dtype, device=device))
    iDFk = weight.unsqueeze(1) * Fk - wXiG @ wwX.T @ Fk

    projection = computeProjectionMatrix(Fk1    =Fk,
                                         Fk2    =iDFk,
                                         data   =data,
                                         S      =None,
                                         dtype  =dtype,
                                         device =device
                                         )
    inverse_square_root_matrix = projection["inverse_square_root_matrix"]
    matrix_JSJ = projection["matrix_JSJ"]
    iDZ = weight.unsqueeze(1) * data - wXiG @ (wwX.T @ data)
    trS = torch.sum(iDZ * data) / num_columns
    ldetD = (
        -Q.shape[0] * torch.log(torch.tensor(lambda_, device=device))
        + logDeterminant(mat = G)
        - logDeterminant(mat = Q)
        - torch.sum(torch.log(weight))
    ).item()

    out = cMLE(Fk                           = Fk,
               num_columns                  = num_columns,
               sample_covariance_trace      = trS,
               inverse_square_root_matrix   = inverse_square_root_matrix,
               matrix_JSJ                   = matrix_JSJ,
               s                            = 0,
               ldet                         = ldetD,
               wSave                        = True,
               onlylogLike                  = False,
               vfixed                       = vfixed,
               dtype                        = dtype,
               device                       = device
               )
    L = out["L"]
    out["s"] = out["v"]
    out.pop("v", None)
    out.pop("L", None)
    if not wSave:
        return out

    iDL = weight.unsqueeze(1) * L - wXiG @ (wwX.T @ L)
    itmp = torch.linalg.solve(
        torch.eye(L.shape[1], dtype=dtype, device=device) + (L.T @ iDL) / out["s"],
        torch.eye(L.shape[1], dtype=dtype, device=device),
    )
    iiLiD = itmp @ (iDL.T / out["s"])
    MFiS11 = (out["M"] @ (iDFk.T / out["s"]) - ((out["M"] @ (iDFk.T / out["s"])) @ L) @ iiLiD)
    out["w"] = MFiS11 @ data
    out["V"] = MFiS11 @ (Fk @ out["M"])
    wlk = wXiG.T @ data - wXiG.T @ L @ (iiLiD @ data)

    out["pinfo"] = {"wlk": wlk,
                    "pick": pick
                    }

    return out

# using in autoFRK
# check = none
def initializeLKnFRK(
    data: torch.Tensor,
    location: torch.Tensor,
    nlevel: int = 3,
    weights: list = None,
    n_neighbor: int = 3,
    nu: int = 1
) -> dict:
    """
    Internal function: initializeLKnFRK

    This function initializes the hierarchical multi-resolution structure for FRK (Fixed Rank Kriging),
    handling missing data, imputing via nearest neighbors, and computing level weights and geometric type.

    Parameters:
        data (torch.Tensor): 
            A tensor of shape (n, T) representing the observed data matrix.
            Each column corresponds to a time point (z[t]), and rows correspond to spatial locations.
            Missing values are allowed (should be represented as `torch.nan`).

        location (torch.Tensor): 
            A tensor of shape (n, d) specifying spatial coordinates for each observation.
            `d` is typically 1 (line), 2 (surface), or 3 (volume).

        nlevel (int, optional, default=3): 
            Number of resolution levels in the multi-resolution basis hierarchy.
            Each level corresponds to a distinct spatial scale used for FRK basis construction.

        weights (torch.Tensor, np.ndarray, or list, optional): 
            An optional weight vector or diagonal weight matrix of length/size `n`.
            If not provided, all weights are set to 1.

        n_neighbor (int, optional, default=3): 
            Number of nearest neighbors used for imputing missing data (fast approximation method).
            If `data` contains missing entries, they are replaced with the average of their `n_neighbor`
            nearest observed values.

        nu (int, optional, default=1): 
            Smoothness parameter controlling the relative contribution of fine and coarse scales.
            Used in the computation of the level weights `alpha`.

    Returns:
        dict
            - **x** (`torch.Tensor`): filtered location matrix after removing empty rows.
            - **z** (`torch.Tensor`): data matrix after imputation and filtering.
            - **n** (`torch.Tensor`): number of valid spatial locations.
            - **alpha** (`torch.Tensor`): normalized weights for each resolution level.
            - **gtype** (`str`): geometry type ("LKInterval", "LKRectangle", or "LKBox") depending on spatial dimension.
            - **weights** (`torch.Tensor`): weight vector for spatial locations.
            - **nlevel** (`int`): number of hierarchical levels.
            - **location** (`torch.Tensor`): original location matrix (possibly with NA rows removed).
            - **pick** (`torch.Tensor`): indices (1-based, to match R) of retained rows from the original data.
    """
    dtype=data.dtype
    device=data.device

    data = data.detach().cpu().numpy()
    location = location.detach().cpu().numpy()

    non_empty = ~np.all(np.isnan(data), axis=0)
    data = data[:, non_empty]

    valid_rows = ~np.all(np.isnan(data), axis=1)
    data = data[valid_rows, :]
    x = location[valid_rows, :]
    pick = np.where(valid_rows)[0]

    # Impute missing values using nearest neighbors
    nas = np.isnan(data).sum()
    if nas > 0:
        for tt in range(data.shape[1]):
            where = np.isnan(data[:, tt])
            if not np.any(where):
                continue
            cidx = np.where(~where)[0]
            nbrs = NearestNeighbors(n_neighbors=n_neighbor).fit(x[cidx, :])
            distances, nn_index = nbrs.kneighbors(x[where, :])
            nn_index = cidx[nn_index]
            nn_values = data[nn_index, tt]
            data[where, tt] = np.nanmean(nn_values, axis=1)

    z = np.asarray(data)
    n, d = x.shape

    if d == 1:
        gtype = "LKInterval"
    elif d == 2:
        gtype = "LKRectangle"
    else:
        gtype = "LKBox"

    thetaL = 2.0 ** (-1 * np.arange(1, nlevel + 1))
    alpha = thetaL ** (2 * nu)
    alpha = alpha / np.sum(alpha)

    if weights is None:
        weights = np.ones(n)

    out = {"x": x,
           "z": z,
           "n": n,
           "alpha": alpha,
           "gtype": gtype,
           "weights": weights,
           "nlevel": nlevel,
           "location": location,
           "pick": pick
           }
    out = to_tensor(out, dtype=dtype, device=device)
    return out


# using in autoFRK
# check = none
def setLKnFRKOption(
    LK_obj: dict,
    Fk: torch.Tensor,
    nc: Optional[torch.Tensor] = None,
    Ks: Optional[int] = None,
    a_wght: Optional[float] = None,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = 'cpu'
) -> dict:
    """
    Internal function: setLKnFRKOption

    Parameters
    ----------
    LK_obj : dict
        A dictionary produced from `initializeLKnFRK()`, containing 'x', 'z', 'alpha', etc.
    Fk : torch.Tensor
        n x K basis function matrix. Each column is a basis function evaluated at locations.
    nc : torch.Tensor, optional
        Numeric matrix/vector produced by `setNC()`. Default is None.
    Ks : int, optional
        Number of basis functions. Default is Fk.shape[1].
    a_wght : float, optional
        Scalar weight used in basis construction. Default is None (will be set to 2*d + 0.01).

    Returns
    -------
    dict
        A dictionary containing:
        - DfromLK : dict with keys Q, weights, wX, G, lambda, pick
        - s : estimated scale parameter
        - LKobj : dictionary containing summary, par.grid, LKinfo.MLE, lnLike.eval, lambda.MLE, call, taskID
    """
    x = LK_obj['x']
    z = LK_obj['z']
    alpha = LK_obj['alpha']
    alpha = alpha / alpha.sum()
    gtype = LK_obj['gtype']
    weights = LK_obj['weights']
    if len(LK_obj['pick']) < len(weights):
        weights = weights[LK_obj['pick']]
    nlevel = LK_obj['nlevel']
    TT = z.shape[1]
    Fk = Fk[LK_obj['pick'], :]

    if nc is None:
        nc = setNC(z,
                   x,
                   nlevel
                   )
    if a_wght is None:
        a_wght = 2 * x.shape[1] + 0.01

    info = LKrigSetup(x         = x,
                      a_wght    = a_wght,
                      nlevel    = nlevel,
                      NC        = nc,
                      alpha     = alpha,
                      LKGeometry= gtype,
                      lambda_   = 1.0
                      )              

    location = x
    phi = LKrig_basis(location,
                      info
                      )
    w = torch.diag(torch.sqrt(weights))
    wX = w @ phi
    wwX = w @ wX
    XwX = wX.T @ wX

    Qini = LKrig_precision(info)

    def iniLike(par,
                data=z,
                full=False
    ) -> Union[dict, torch.Tensor]:
        """
        inner function ...
        """
        lambda_ = torch.exp(torch.tensor(par, dtype=dtype, device=device))
        G = XwX + lambda_ * Qini
        wXiG = wwX @ torch.linalg.inv(G)
        iDFk = weights * Fk - wXiG @ (wwX.T @ Fk)
        iDZ = weights * data - wXiG @ (wwX.T @ data)
        ldetD = -Qini.shape[0] * torch.log(lambda_) + logDeterminant(mat = G_mat)
        trS = torch.sum(iDZ * data) / TT
        half = getInverseSquareRootMatrix(Fk, iDFk)
        ihFiD = half @ iDFk.T
        LSL = (ihFiD @ data) @ (ihFiD @ data).T / TT
        if not full:
            return cMLE(Fk,
                        TT,
                        trS,
                        half,
                        LSL,
                        s=0,
                        ldet=ldetD,
                        wSave=False
                        )['negloglik']
        else:
            llike = ldetD - logDeterminant(mat = Qini) - torch.log(weights).sum()
            return cMLE(Fk,
                        TT,
                        trS,
                        half,
                        LSL,
                        s=0,
                        ldet=llike,
                        wSave=True,
                        onlylogLike=False,
                        vfixed=None
                        )

    sol = minimize_scalar(iniLike,
                          bounds=(-16, 16),
                          method="bounded",
                          options={"xatol": np.finfo(float).eps ** 0.025}
                          )
    lambda_MLE = to_tensor(sol.x, dtype=dtype, device=device)
    out = iniLike(to_tensor(sol.x, dtype=dtype, device=device),
                  z,
                  full=True
                  )
    llike = out['negloglik']
    info_MLE = LKrigSetup(x=x,
                          a_wght=a_wght,
                          nlevel=nlevel,
                          NC=nc,
                          alpha=alpha,
                          LKGeometry=gtype,
                          lambda_=lambda_MLE
                          )
    info_MLE['llike'] = llike
    info_MLE['time'] = None
    Q = LKrig_precision(info_MLE)
    G_mat = wX.T @ wX + info_MLE['lambda_'] * Q

    ret =   {
                "DfromLK": {
                    "Q": Q,
                    "weights": weights,
                    "wX": wX,
                    "G": G_mat,
                    "lambda": info_MLE['lambda_'],
                    "pick": LK_obj['pick']
                },
                "s": out['v'],
                "LKobj": {
                    "summary": None,
                    "par_grid": None,
                    "LKinfo_MLE": info_MLE,
                    "lnLike_eval": None,
                    "lambda_MLE": info_MLE['lambda_'],
                    "call": None,
                    "taskID": None
                }
            }
    return to_tensor(ret, dtype=dtype, device=device)




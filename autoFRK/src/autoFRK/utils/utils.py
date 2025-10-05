"""
Title: Setup file of autoFRK-Python Project
Author: Hsu, Yao-Chih
Version: 1140807
Reference:
"""

# development only
import os
import sys
sys.path.append(os.path.abspath("./src"))

# import modules
import tempfile
import os
import shutil
import torch
import numpy as np
import faiss
import gc
from typing import Optional, Union
from sklearn.neighbors import NearestNeighbors
from autoFRK.utils.logger import setup_logger
from autoFRK.mrts import MRTS

# logger config
LOGGER = setup_logger()

# fast mode KNN for missing data imputation, using in autoFRK
# Its have OpenMP issue, set environment variable OMP_NUM_THREADS=1 to avoid it, or use sklearn version below
# check = ok
def fast_mode_knn(
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
    loc = loc.detach().cpu().numpy().astype(np.float32)

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
    device: Optional[Union[torch.device, str]] = 'cpu'
) -> torch.Tensor:
    # 去除全為 NaN 的欄位
    not_all_nan = ~torch.isnan(data).all(dim=0)
    data = data[:, not_all_nan]

    # 檢查資料中是否有缺失值
    is_data_with_missing_values = torch.isnan(data).any()

    # 找出整行都是 NaN 的列（完全缺失）
    na_rows = torch.isnan(data).all(dim=1)
    pick = torch.arange(data.shape[0])
    if na_rows.any():
        data = data[~na_rows]
        loc = loc[~na_rows]  # 同步刪除 loc 中相同的行 need fix
        D = D[~na_rows][:, ~na_rows]
        pick = pick[~na_rows]
        is_data_with_missing_values = torch.isnan(data).any()

    # 如果 D 未提供，則初始化為單位對角矩陣
    if D is None:
        D = torch.eye(data.shape[0], device=data.device)

    # 取得位置維度
    d = loc.shape[1]

    # 計算 klim 與選 knot
    N = len(pick)
    klim = int(min(N, np.round(10 * np.sqrt(N))))
    if N < max_knot:
        knot = loc[pick, :]
    else:
        knot = subKnot(x=loc[pick, :],
                       nknot=min(max_knot, klim),
                       device=device
                       ).to(device=device)

    # 處理 K 值
    if max_rank is not None:
        max_rank = round(max_rank)
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
        K = torch.arange(d + 1, max_rank, step).round().to(torch.int).unique()
        if len(K) > 30:
            K = torch.linspace(d + 1, max_rank, 30).round().to(torch.int).unique()

    # Fk 為 None 時初始化 basis function 值
    if Fk is None:
        mrts = MRTS(locs=loc, k=max(K), device=device)  # 待修 (knot, max(K), loc, max_knot) need fix
        Fk = mrts.forward()

    AIC_list = [float('inf')] * len(K)
    num_data_columns = data.shape[1]

    if method == "EM" and DfromLK is None:
        for k in range(len(K)):
            AIC_list[k] = indeMLE(data,
                                  Fk[pick, :K[k]],
                                  D,
                                  maxit,
                                  avgtol,
                                  wSave=False,
                                  verbose=False
                                  )["negloglik"]
    else:
        if is_data_with_missing_values:
            data = fast_mode_knn_sklearn(data=data, loc=loc, n_neighbor=num_neighbors) 
        if DfromLK is None:
            iD = torch.linalg.solve(D, torch.eye(D.shape[0], device=D.device))
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
            inverse_square_root_matrix = get_inverse_square_root_matrix(Fk_k, iDFk_k)
            ihFiD = inverse_square_root_matrix @ iDFk_k.T
            tmp = torch.matmul(ihFiD, data)
            matrix_JSJ = torch.matmul(tmp, tmp.T) / num_data_columns
            matrix_JSJ = (matrix_JSJ + matrix_JSJ.T) / 2
            AIC_list[k] = cMLE(Fk=Fk_k,
                               num_columns=num_data_columns,
                               sample_covariance_trace=sample_covariance_trace,
                               inverse_square_root_matrix=inverse_square_root_matrix,
                               matrix_JSJ=matrix_JSJ
                               )["negloglik"]

    # 計算 AIC 並選出最佳 K 值
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
def get_inverse_square_root_matrix(left_matrix, right_matrix):
    mat = left_matrix.T @ right_matrix  # A^T * B
    mat = (mat + mat.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(mat)
    inv_sqrt_eigvals = torch.diag(torch.clamp(eigvals, min=1e-10).rsqrt())
    return eigvecs @ inv_sqrt_eigvals @ eigvecs.T

# subset knot selection for autoFRK, using in selectBasis
# check = ok
def subKnot(
    x: torch.Tensor, 
    nknot: int, 
    xrng: torch.Tensor = None, 
    nsamp: int = 1, 
    device: Optional[Union[torch.device, str]]='cpu'
) -> torch.Tensor:
    x = x.to(device)
    x = torch.sort(x, dim=0).values
    xdim = x.shape  # (N, D)

    if xrng is None:
        xrng = torch.stack([x.min(dim=0).values, x.max(dim=0).values], dim=0)

    rng = torch.sqrt(xrng[1] - xrng[0])
    if (rng == 0).any():
        rng[rng == 0] = rng[rng > 0].min() / 5
    rng = rng * 10 / rng.min()
    rng_max_index = torch.argmax(rng).item()

    log_rng = torch.log(rng)
    nmbin = torch.round(torch.exp(log_rng * torch.log(torch.tensor(nknot, dtype=torch.float32)) / log_rng.sum())).int()
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
                brk = torch.tensor(np.quantile(x[:, kk].cpu().numpy(), np.linspace(0, 1, nmbin[kk] + 1)), device=device)
                brk[0] -= 1e-8
                grp = torch.bucketize(x[:, kk], brk) - 1
            gvec += kconst * grp
            kconst *= nmbin[kk]

        cnt += 1

    gvec_np = gvec.cpu().numpy()
    index = []
    for g in np.unique(gvec_np):
        idx = np.where(gvec_np == g)[0]
        if len(idx) == 1:
            index.append(idx[0])
        else:
            np.random.seed(int(np.mean(idx)))
            index.extend(np.random.choice(idx, size=min(nsamp, len(idx)), replace=False))

    index = torch.tensor(index, device=device)
    return x[index]

# compute negative log likelihood for autoFRK, using in selectBasis
# check = none
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
    device: Optional[Union[torch.device, str]] = 'cpu'
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
        nrow_Fk=nrow_Fk,
        ncol_Fk=Fk.shape[1],
        s=s,
        p=num_columns,
        matrix_JSJ=matrix_JSJ,
        sample_covariance_trace=sample_covariance_trace,
        vfixed=vfixed,
        ldet=ldet,
        device=device
    )

    negative_log_likelihood = likelihood_object['negative_log_likelihood']

    if onlylogLike:
        return {'negloglik': negative_log_likelihood}

    P = likelihood_object['P']
    d_hat = likelihood_object['d_hat']
    v = likelihood_object['v']
    M = inverse_square_root_matrix @ P @ (d_hat * P.T) @ inverse_square_root_matrix

    if not wSave:
        L = None
    elif d_hat[0] != 0:
        L = Fk @ ((torch.sqrt(d_hat) * P.T) @ inverse_square_root_matrix)
        L = L[:, d_hat > 0]
    else:
        L = torch.zeros((nrow_Fk, 1), dtype=Fk.dtype, device=Fk.device)

    return {
        'v': v,
        'M': M,
        's': s,
        'negloglik': negative_log_likelihood,
        'L': L
    }

# compute negative log likelihood for autoFRK, using in cMLE
# check = none
def computeNegativeLikelihood(
    nrow_Fk: int,
    ncol_Fk: int,
    s: int,
    p: int,
    matrix_JSJ: torch.Tensor,
    sample_covariance_trace: float,
    vfixed: float = None,
    ldet: float = 0.0,
    device: Optional[Union[torch.device, str]]='cpu'
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
    matrix_JSJ = matrix_JSJ.to(device)

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
        v = estimateV(d=eigenvalues_JSJ, 
                      s=s, 
                      sample_covariance_trace=sample_covariance_trace, 
                      n=nrow_Fk
                      )
    else:
        v = vfixed

    d = torch.clamp(eigenvalues_JSJ, min=0)
    d_hat = estimateEta(d, s, v)

    negative_log_likelihood = neg2llik(d=d, 
                                       s=s, 
                                       v=v, 
                                       sample_covariance_trace=sample_covariance_trace, 
                                       sample_size=nrow_Fk
                                       ) * p + ldet * p

    return {
        "negative_log_likelihood": negative_log_likelihood,
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
    n: int
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
    ks = torch.arange(1, k + 1, device=d.device)
    if k == n:
        ks[-1] = n - 1

    eligible_indices = torch.nonzero(d > (sample_covariance_trace - cumulative_d_values) / (n - ks)).flatten()
    L = int(torch.max(eligible_indices))
    if L >= n:
        L = n - 1

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
# check = none
def neg2llik(
    d: torch.Tensor,
    s: float,
    v: float,
    sample_covariance_trace: float,
    sample_size: int
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
    eta = estimateEta(d, s, v)

    if torch.max(eta / (s + v)) > 1e20:
        return float("inf")
    
    log_det_term = torch.sum(torch.log(eta + s + v))
    log_sv_term = torch.log(s + v) * (sample_size - k)
    trace_term = sample_covariance_trace / (s + v)
    eta_term = torch.sum(d * eta / (eta + s + v)) / (s + v)

    return sample_size * torch.log(2 * torch.pi) + log_det_term + log_sv_term + trace_term - eta_term

# independent maximum likelihood estimation for autoFRK, using in selectBasis
# check = none
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
    device: Optional[Union[torch.device, str]]='cpu'
) -> dict:
    """
    
    """
    device = torch.device(device)
    data = data.to(device)
    Fk = Fk.to(device)

    withNA = torch.isnan(data).any().item()

    TT = data.shape[1]
    empty = torch.isnan(data).all(dim=0)
    notempty = (~empty).nonzero(as_tuple=True)[0]
    if empty.any():
        data = data[:, notempty]

    del_rows = torch.isnan(data).all(dim=1).nonzero(as_tuple=True)[0]
    pick = torch.arange(data.shape[0], device=device)

    if D is None:
        D = torch.eye(data.shape[0], device=device).to_sparse()

    if not torch.allclose(D, torch.diag(torch.diagonal(D))):
        D0 = toSparseMatrix(mat=D)
    else:
        D0 = torch.diag(torch.diag(D)).to_sparse()

    if withNA and len(del_rows) > 0:
        pick = pick[~torch.isin(pick, del_rows)]
        data = data[~torch.isin(torch.arange(data.shape[0], device=device), del_rows), :]
        Fk = Fk[~torch.isin(torch.arange(Fk.shape[0], device=device), del_rows), :]
        if not torch.allclose(D, torch.diag(torch.diagonal(D))):
            D = D[~torch.isin(torch.arange(D.shape[0], device=device), del_rows)][:, ~torch.isin(torch.arange(D.shape[1], device=device), del_rows)]
        else:
            keep_mask = ~torch.isin(torch.arange(D.shape[0], device=device), del_rows)
            full_diag = torch.zeros(D.shape[0], device=device)
            full_diag[keep_mask] = torch.diagonal(D)[keep_mask]
            D = torch.diag(full_diag)
        withNA = torch.isnan(data).any().item()

    N = data.shape[0]
    K = Fk.shape[1]
    Depsilon = toSparseMatrix(mat=D)
    is_diag = torch.allclose(D, torch.diag(torch.diagonal(D)))
    mean_diag = torch.mean(torch.diagonal(D))
    isimat = is_diag and torch.allclose(torch.diagonal(Depsilon), mean_diag.repeat(N), atol=1e-10)

    if not withNA:
        if isimat and DfromLK is None:
            sigma = 0  # we cannot find `.Option$sigma_FRK` in the R code # need fix
            out = cMLEimat(Fk, 
                           data, 
                           s=sigma, 
                           wSave=wSave
                           )
            if out['v'] is not None:
                out['s'] = out['v'] if sigma == 0 else sigma
                del out['v']
            if wSave:
                w = torch.zeros((K, TT), device=device)
                w[:, notempty] = out['w']
                out['w'] = w
                out['pinfo'] = {'D': D0, 
                                'pick': pick
                                }
            return out
        elif DfromLK is None:
            out = cMLEsp(Fk, 
                         data, 
                         Depsilon, 
                         wSave
                         )
            if wSave:
                w = torch.zeros((K, TT), device=device)
                w[:, notempty] = out['w']
                out['w'] = w
                out['pinfo'] = {'D': D0, 
                                'pick': pick
                                }
            return out
        else:
            out = cMLElk(Fk, 
                         data, 
                         Depsilon, 
                         wSave, 
                         DfromLK, 
                         vfixed
                         )
            if wSave:
                w = torch.zeros((K, TT), device=device)
                w[:, notempty] = out['w']
                out['w'] = w
            return out
    else:
        out = EM0miss(Fk, 
                      data, 
                      Depsilon, 
                      maxit, 
                      avgtol, 
                      wSave, 
                      external=False, 
                      DfromLK=DfromLK, 
                      vfixed=vfixed, 
                      verbose=verbose
                      )
        if wSave:
            w = torch.zeros((K, TT), device=device)
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
# check = none
def cMLEimat(
    Fk: torch.Tensor,
    data: torch.Tensor,
    s: float,
    wSave: bool = False,
    S: Optional[torch.Tensor] = None,
    onlylogLike: Optional[bool] = None,
    device: Optional[Union[torch.device, str]]='cpu'
) -> dict:

    if onlylogLike is None:
        onlylogLike = not wSave

    Fk = Fk.to(device)
    data = data.to(device)

    nrow_Fk, ncol_Fk = Fk.shape
    num_columns = data.shape[1]

    projection = computeProjectionMatrix(Fk1=Fk, 
                                         Fk2=Fk, 
                                         data=data, 
                                         S=S, 
                                         device=device
                                         )
    inverse_square_root_matrix = projection["inverse_square_root_matrix"]
    matrix_JSJ = projection["matrix_JSJ"]

    sample_covariance_trace = torch.sum(data ** 2) / num_columns

    likelihood_object = computeNegativeLikelihood(nrow_Fk=nrow_Fk,
                                                  ncol_Fk=ncol_Fk,
                                                  s=s,
                                                  p=num_columns,
                                                  matrix_JSJ=matrix_JSJ,
                                                  sample_covariance_trace=sample_covariance_trace,
                                                  device=device
                                                  )

    negative_log_likelihood = likelihood_object["negative_log_likelihood"]

    if onlylogLike:
        return {"negloglik": negative_log_likelihood}

    P = likelihood_object["P"]
    d_hat = likelihood_object["d_hat"]
    v = likelihood_object["v"]

    M = inverse_square_root_matrix @ P @ (d_hat * P.T) @ inverse_square_root_matrix

    if not wSave:
        return {"v": v, 
                "M": M, 
                "s": s, 
                "negloglik": negative_log_likelihood
                }

    L = Fk @ ((torch.sqrt(d_hat) * P.T) @ inverse_square_root_matrix).T

    if ncol_Fk > 2:
        reduced_columns = torch.cat([
            torch.tensor([0], device=device),
            (d_hat[1:ncol_Fk] > 0).nonzero(as_tuple=True)[0]
        ])
    else:
        reduced_columns = torch.tensor([ncol_Fk - 1], device=device)

    L = L[:, reduced_columns]

    invD = torch.ones(nrow_Fk, device=device) / (s + v)
    iDZ = invD[:, None] * data

    right = L @ (torch.linalg.inv(torch.eye(L.shape[1], device=device) + L.T @ (invD[:, None] * L)) @ (L.T @ iDZ))

    INVtZ = iDZ - invD[:, None] * right
    etatt = M @ Fk.T @ INVtZ

    GM = Fk @ M

    diag_matrix = (s + v) * torch.eye(nrow_Fk, device=device)
    V = M - GM.T @ invCz(R=diag_matrix,
                         L=L, 
                         z=GM,
                         device=device
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
    device: Optional[Union[torch.device, str]]='cpu'
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
    Fk1 = Fk1.to(device)
    Fk2 = Fk2.to(device)
    data = data.to(device)
    if S is not None:
        S = S.to(device)

    num_columns = data.shape[1]
    inverse_square_root_matrix = getInverseSquareRootMatrix(A=Fk1, 
                                                            B=Fk2, 
                                                            device=device
                                                            )
    inverse_square_root_on_Fk2 = inverse_square_root_matrix @ Fk2.T

    if S is None:
        matrix_JSJ = (inverse_square_root_on_Fk2 @ data) @ (inverse_square_root_on_Fk2 @ data).T / num_columns
    else:
        matrix_JSJ = (inverse_square_root_on_Fk2 @ S) @ inverse_square_root_on_Fk2.T

    matrix_JSJ = (matrix_JSJ + matrix_JSJ.T) / 2

    return {
        "inverse_square_root_matrix": inverse_square_root_matrix,
        "matrix_JSJ": matrix_JSJ
    }

# using in computeProjectionMatrix
# check = ok
def getInverseSquareRootMatrix(
    A: torch.Tensor, 
    B: torch.Tensor, 
    device: Optional[Union[torch.device, str]]='cpu',
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
    A = A.to(device)
    B = B.to(device)

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
    device: Optional[Union[torch.device, str]]='cpu'
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
    R = R.to(device)
    L = L.to(device)
    z = z.to(device)

    if z.dim() == 1:
        z = z.unsqueeze(1)

    K = L.shape[1]
    iR = torch.linalg.pinv(R)
    iRZ = iR @ z
    right = L @ torch.linalg.inv(torch.eye(K, device=device) + (L.T @ iR @ L)) @ (L.T @ iRZ) 
    result = iRZ - iR @ right

    return result.T

# using in indeMLE
# check = none
def EM0miss(
    Fk: torch.Tensor, 
    data: torch.Tensor, 
    Depsilon: torch.Tensor, 
    maxit: int=100, 
    avgtol: float=1e-4, 
    wSave: bool=False, 
    external: bool=False,
    DfromLK: dict=None,
    vfixed: float=None,
    verbose: bool=True,
    device: Optional[Union[torch.device, str]] = 'cpu'
) -> dict:
    """

    """
    Fk = Fk.to(device)
    data = data.to(device)
    Depsilon = Depsilon.to(device)

    O = ~torch.isnan(data)
    TT = data.shape[1]
    ncol_Fk = Fk.shape[1]
    tmpdir = tempfile.mkdtemp()
    oldfile = os.path.join(tmpdir, "old_par.pt")

    ziDz = torch.full((TT,), float('nan'), device=device)
    ziDB = torch.full((TT, ncol_Fk), float('nan'), device=device)
    db = {}
    D = Depsilon
    iD = torch.linalg.inv(D)
    diagD = isDiagonal(D)

    if DfromLK is not None:
        pick = DfromLK.get("pick", None)
        weights = torch.tensor(DfromLK["weights"], device=device)
        if pick is None:
            pick = torch.arange(len(weights), device=device)
        else:
            pick = torch.tensor(pick, dtype=torch.long, device=device)
        weight = weights[pick]

        DfromLK_wX = DfromLK["wX"]
        if not torch.is_tensor(DfromLK_wX):
            DfromLK_wX = torch.tensor(DfromLK_wX, device=device)
        DfromLK_wX = DfromLK_wX[pick, :].clone().detach()

        DfromLK_Q = DfromLK["Q"]
        if not torch.is_tensor(DfromLK_Q):
            DfromLK_Q = torch.tensor(DfromLK_Q, device=device)

        wwX = torch.diag(torch.sqrt(weight)) @ DfromLK_wX
        lQ = DfromLK["lambda"] * DfromLK_Q.clone().detach()

    for tt in range(TT):
        if DfromLK is not None:
            obs_idx = O[:, tt].bool()
            iDt = None
            if obs_idx.sum() == O.shape[0]:
                G_inv = torch.linalg.inv(DfromLK["G"].to(device=device))
                wXiG = wwX @ G_inv
            else:
                wX_obs = DfromLK_wX[obs_idx, :].to(device)
                G = wX_obs.T @ wX_obs + lQ.to(device)
                wXiG = wwX[obs_idx, :] @ torch.linalg.inv(G)

            Bt = Fk[obs_idx, :].to(device)
            if Bt.ndim == 1:
                Bt = Bt.unsqueeze(0)

            iDBt = weight[obs_idx].unsqueeze(1) * Bt - wXiG @ (wwX[obs_idx, :].T @ Bt)
            zt = data[obs_idx, tt].to(device=device)
            ziDz[tt] = torch.sum(zt * (weight[obs_idx] * zt - wXiG @ (wwX[obs_idx, :].T @ zt)))
            ziDB[tt, :] = (zt @ iDBt).squeeze()
            BiDBt = Bt.T @ iDBt

        else:
            if not diagD:
                iDt = torch.linalg.inv(D[obs_idx][:, obs_idx].to(device))
            else:
                iDt = iD[obs_idx][:, obs_idx].to(device)

            Bt = Fk[obs_idx, :].to(device)
            if Bt.ndim == 1:
                Bt = Bt.unsqueeze(0)

            iDBt = iDt @ Bt
            zt = data[obs_idx, tt]
            ziDz[tt] = torch.sum(zt * (iDt @ zt))
            ziDB[tt, :] = (zt @ iDBt).squeeze()
            BiDBt = Bt.T @ iDBt

        db[tt] = {
            "iDBt": iDBt,
            "zt": zt,
            "BiDBt": BiDBt,
            "external": external,
            "oldfile": oldfile,
        }

    del iDt, Bt, iDBt, zt, BiDBt
    gc.collect()
    torch.cuda.empty_cache()

    dif = float("inf")
    cnt = 0
    Z0 = data.clone()
    Z0[torch.isnan(Z0)] = 0
    old = cMLEimat(Fk=Fk, 
                   data=Z0, 
                   s=0, 
                   wSave=True,
                   device=device
                   )
    if vfixed is None:
        old["s"] = old["v"]
    else:
        old["s"] = vfixed.to(old["v"].device)
    old["M"] = convertToPositiveDefinite(old["M"])
    Ptt1 = old["M"]
    if external:
        torch.save({"old": old, "Ptt1": Ptt1}, oldfile)

    while (dif > (avgtol * (100 * (ncol_Fk ** 2)))) and (cnt < maxit):
        etatt = torch.zeros((ncol_Fk, TT), device=device)
        sumPtt = torch.zeros((ncol_Fk, ncol_Fk), device=device)
        s1 = torch.zeros(TT, device=device)

        if external:
            saved = torch.load(oldfile)
            old = saved["old"]
            Ptt1 = saved["Ptt1"]

        for tt in range(TT):
            iDBt = db[tt]["iDBt"].to(device)
            zt = db[tt]["zt"].to(device)
            BiDBt = db[tt]["BiDBt"].to(device)
            ginv_Ptt1 = torch.linalg.pinv(convertToPositiveDefinite(Ptt1))
            iP = convertToPositiveDefinite(ginv_Ptt1 + BiDBt / old["s"])
            Ptt = torch.linalg.inv(iP)
            Gt = (Ptt @ iDBt.T) / old["s"]
            eta = Gt @ zt
            s1kk = torch.diagonal(BiDBt @ (eta.unsqueeze(1) @ eta.unsqueeze(0) + Ptt))

            sumPtt += Ptt
            etatt[:, tt] = eta
            s1[tt] = torch.sum(s1kk)

        if vfixed is None:
            s = torch.max(
                (torch.sum(ziDz) - 2 * torch.sum(ziDB * etatt.T) + torch.sum(s1)) / torch.sum(O),
                torch.tensor(1e-8, device=ziDz.device)
            )
            new = {
                "M": (etatt @ etatt.T + sumPtt) / TT,
                "s": s,
            }
        else:
            new = {
                "M": (etatt @ etatt.T + sumPtt) / TT,
                "s": vfixed.to(device),
            }

        new["M"] = (new["M"] + new["M"].T) / 2
        dif = torch.sum(torch.abs(new["M"] - old["M"])) + torch.abs(new["s"] - old["s"])
        cnt += 1
        old = new
        Ptt1 = old["M"]

        if external:
            torch.save({"old": old, "Ptt1": Ptt1}, oldfile)

    if verbose:
        info_msg = f'Number of iteration: {cnt}'
        LOGGER.info(info_msg)
    shutil.rmtree(tmpdir, ignore_errors=True)
    n2loglik = computeLikelihood(data=data,
                                 Fk=Fk,
                                 M=new["M"],
                                 S=new["s"],
                                 Depsilon=Depsilon,
                                 device=device
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
                wXiG = wwX @ torch.linalg.solve(DfromLK["G"], torch.eye(DfromLK["G"].shape[0], device=device))
            else:
                wX_tt = DfromLK_wX[obs_idx]
                G = wX_tt.T @ wX_tt + lQ
                wXiG = wwX[obs_idx] @ torch.linalg.solve(G, torch.eye(G.shape[0], device=device))

            dat = data[obs_idx, tt]
            Lt = L[obs_idx]
            iDL = weight[obs_idx].unsqueeze(1) * Lt - wXiG @ (wwX[obs_idx].T @ Lt)
            itmp = torch.linalg.solve(
                torch.eye(L.shape[1], device=device) + (Lt.T @ iDL) / out["s"],
                torch.eye(L.shape[1], device=device)
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
    device: Optional[Union[torch.device, str]] = 'cpu'
) -> torch.Tensor:
    """
    Internal function: convert a matrix to positive definite

    Parameters:
        mat (torch.Tensor): Input 2D matrix (square, symmetric or not).
        device (torch.device): Device to perform computation on ("cpu" or "cuda").

    Returns:
        torch.Tensor: A positive-definite version of the input matrix.
    """
    mat = mat.to(device)

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
        mat = mat + torch.eye(mat.shape[0], device=device) * adjustment

    return mat

# using in EM0miss
# check = ok
def computeLikelihood(
    data: torch.Tensor,
    Fk: torch.Tensor,
    M: torch.Tensor,
    s: float,
    Depsilon: torch.Tensor,
    device: Optional[Union[torch.device, str]] = 'cpu'
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
    data = data.to(device)
    Fk = Fk.to(device)
    M = M.to(device)
    Depsilon = Depsilon.to(device)

    non_missing_points_matrix = ~torch.isnan(data)
    num_columns = data.shape[1]

    n2loglik = non_missing_points_matrix.sum() * torch.log(torch.tensor(2 * torch.pi, device=device))
    R = s * Depsilon
    eg = eigenDecompose(M,
                        device=device
                        )
    K = Fk.shape[1]
    L = Fk @ eg["vector"] @ torch.diag(torch.sqrt(torch.clamp(eg["value"], min=0.0))) @ eg["vector"].T
    
    for t in range(num_columns):
        mask = non_missing_points_matrix[:, t]
        zt = data[mask, t]

        # skip all-missing column
        if zt.numel() == 0:
            continue

        Rt = R[mask][:, mask]
        Lt = L[mask]

        log_det = calculateLogDeterminant(Rt, 
                                          Lt, 
                                          K, 
                                          device=device
                                          )
        inv_cz_val = invCz(Rt, 
                           Lt, 
                           zt, 
                           device=device
                           )
        n2loglik += log_det + torch.sum(zt * inv_cz_val)

    return n2loglik.item()

# using in computeLikelihood
# check = ok
def eigenDecompose(
    matrix: torch.Tensor,
    device: Optional[Union[torch.device, str]] = 'cpu'
) -> dict:
    """
    Internal function: Eigen-decompose a matrix

    Parameters:
        matrix (torch.Tensor): (K x K) symmetric matrix
        device (str or torch.device): computation device

    Returns:
        dict with keys:
            'value': (K,) tensor of eigenvalues
            'vector': (K x K) tensor of eigenvectors (columns)
    """
    matrix = matrix.to(device)

    # Use symmetric eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

    return {
        'value': eigenvalues,
        'vector': eigenvectors
    }

# using in computeLikelihood
# check = ok
def calculateLogDeterminant(
    R: torch.Tensor,
    L: torch.Tensor,
    K: int=None,
    device: Optional[Union[str, torch.device]] = 'cpu'
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
    R = R.to(device)
    L = L.to(device)

    if K is None:
        K = L.shape[1]

    first_part_determinant = torch.logdet(torch.eye(K, device=device) + L.T @ torch.linalg.solve(R, L))
    second_part_determinant = torch.logdet(R)

    return (first_part_determinant + second_part_determinant).item()

# using in indeMLE














































































































































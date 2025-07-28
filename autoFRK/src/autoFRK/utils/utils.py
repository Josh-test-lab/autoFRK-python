import torch
import numpy as np
import faiss
from typing import Optional, Union
from sklearn.neighbors import NearestNeighbors
from autoFRK.utils.logger import setup_logger
from autoFRK.mrts import MRTS


# logger config
LOGGER = setup_logger()

# fast mode KNN for missing data imputation, using in autoFRK
# Its have OpenMP issue, set environment variable OMP_NUM_THREADS=1 to avoid it, or use sklearn version below
def fast_mode_knn(data: torch.Tensor,
                  loc: torch.Tensor, 
                  n_neighbor: int = 3
                  ) -> torch.Tensor:
    """
    The fast mode for autoFRK by using KNN for missing data imputation.

    Args:
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
def fast_mode_knn_sklearn(data: torch.Tensor,
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
def selectBasis(data: torch.Tensor,
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
        knot = subKnot(loc[pick, :], min(max_knot, klim))

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
            AIC_list[k] = cMLE(
                Fk=Fk_k,
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

def get_inverse_square_root_matrix(left_matrix, right_matrix):
    mat = left_matrix.T @ right_matrix  # A^T * B
    mat = (mat + mat.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(mat)
    inv_sqrt_eigvals = torch.diag(torch.clamp(eigvals, min=1e-10).rsqrt())
    return eigvecs @ inv_sqrt_eigvals @ eigvecs.T



























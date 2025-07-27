import torch
import numpy as np
import faiss
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

        # skip if low known values
        if len(known_idx) < n_neighbor:
            #LOGGER.warning(f'Column {tt} has too few known values to impute ({len(known_idx)} < {n_neighbor}).')
            continue

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

        if len(known_idx) < n_neighbor:
            print(f'Column {tt} has too few known values to impute ({len(known_idx)} < {n_neighbor}).')
            continue

        knn = NearestNeighbors(n_neighbors=n_neighbor, algorithm='auto').fit(loc[known_idx])
        distances, knn_idx = knn.kneighbors(loc[unknown_idx])

        neighbor_vals = col[known_idx[knn_idx]]
        col[where] = np.nanmean(neighbor_vals, axis=1)
        data[:, tt] = col

    return torch.tensor(data, dtype=dtype, device=device)

# select basis function for autoFRK, using in autoFRK
def select_basis(data: torch.Tensor,
                 loc: torch.Tensor,
                 D: torch.Tensor = None,
                 maxit: int = 50,
                 avgtol: float = 1e-6,
                 max_rank: int = None,
                 sequence_rank: list = None,
                 method: str = "fast",
                 num_neighbors: int = 3,
                 max_knot: int = 5000,
                 DfromLK: dict = None,
                 Fk: torch.Tensor = None
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
        D = D[~na_rows][:, ~na_rows]
        pick = pick[~na_rows]
        is_data_with_missing_values = torch.isnan(data).any()

    # 如果 D 未提供，則初始化為單位對角矩陣
    if D is None:
        D = torch.eye(data.shape[0], device=data.device)

    # 取得位置維度
    d = loc.shape[1]




    # 計算 klim 與選 knot
    N = pick.shape[0]
    klim = min(N, round(10 * N ** 0.5))
    if N < max_knot:
        knot = loc
    else:
        knot = loc[torch.randperm(N)[:min(max_knot, klim)]]  # PyTorch 替代 subKnot

    # 處理 K 值
    if max_rank is None:
        if sequence_rank is not None:
            max_rank = round(max(sequence_rank))
        else:
            max_rank = klim

    if sequence_rank is not None:
        K = torch.unique(torch.tensor(sequence_rank, dtype=torch.int)).tolist()
        K = [k for k in K if k > d]
    else:
        K = list(torch.unique(torch.round(
            torch.linspace(d + 1, max_rank, steps=min(30, max_rank - d))
        ).int()).tolist())

    # Fk 為 None 時初始化 basis function 值
    if Fk is None:
        Fk = mrts(knot, max(K), loc, max_knot)  # 假設已實作 mrts()

    AIC_list = [float('inf')] * len(K)
    num_data_columns = data.shape[1]

    if method == "EM" and DfromLK is None:
        for i, k in enumerate(K):
            AIC_list[i] = indeMLE(
                data,
                Fk[pick, :k],
                D,
                maxit,
                avgtol,
                wSave=False,
                verbose=False
            )["negloglik"]
    else:
        if is_data_with_missing_values:
            # 使用 fast_mode_knn 補齊缺失值
            data = fast_mode_knn(data, loc, n_neighbor=num_neighbors)

        if DfromLK is None:
            iD = torch.linalg.inv(D)
            iDFk = iD @ Fk[pick]
            iDZ = iD @ data
        else:
            # 這段需要仿造原 R 中 LK 模型處理，略寫
            raise NotImplementedError("DfromLK 處理尚未實作")

        sample_cov_trace = torch.sum(iDZ * data) / num_data_columns

        for i, k in enumerate(K):
            ihFiD = get_inverse_square_root_matrix(Fk[pick, :k], iDFk[:, :k]) @ iDFk[:, :k].T
            JSJ = ihFiD @ data
            JSJ = (JSJ @ JSJ.T) / num_data_columns
            JSJ = (JSJ + JSJ.T) / 2  # 保證對稱

            AIC_list[i] = cMLE(
                Fk=Fk[pick, :k],
                num_columns=num_data_columns,
                sample_covariance_trace=sample_cov_trace,
                inverse_square_root_matrix=get_inverse_square_root_matrix(Fk[pick, :k], iDFk[:, :k]),
                matrix_JSJ=JSJ
            )["negloglik"]

    # 計算 AIC 並選出最佳 K 值
    df = [
        (k * (k + 1) / 2 + 1 if k <= num_data_columns else k * num_data_columns + 1 - num_data_columns * (num_data_columns - 1) / 2)
        for k in K
    ]
    AIC_list = [aic + 2 * dfi for aic, dfi in zip(AIC_list, df)]
    Kopt = K[int(torch.tensor(AIC_list).argmin())]

    out = Fk[:, :Kopt]
    return out  # 可額外附上屬性作為 mrts 物件





























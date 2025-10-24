import torch
from typing import Union, Dict

def spherical_kernel(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor | None = None,
    lon2: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = "cpu",
) -> torch.Tensor:
    """
    Compute spherical kernel values or kernel matrix between geographic coordinates.

    This unified version combines the functionality of the original `spherical_kernel`
    (formerly `Kf`) and `K`. It computes a thin-plate spline–like spherical kernel 
    based on great-circle distances between latitude–longitude pairs.

    Depending on the inputs, it supports both single-pair and full-matrix computations:
    - If lat1, lon1, lat2, lon2 are scalars → returns a single kernel value.
    - If lat1, lon1, lat2, lon2 are 1D tensors → returns a kernel matrix.

    Parameters
    ----------
    lat1 : torch.Tensor
        Tensor of latitudes (in degrees) for the first set of locations, shape (n,).
    lon1 : torch.Tensor
        Tensor of longitudes (in degrees) for the first set of locations, shape (n,).
    lat2 : torch.Tensor, optional
        Tensor of latitudes (in degrees) for the second set of locations. 
        If None, `lat2 = lat1` is assumed.
    lon2 : torch.Tensor, optional
        Tensor of longitudes (in degrees) for the second set of locations. 
        If None, `lon2 = lon1` is assumed.
    dtype : torch.dtype, optional
        Desired torch dtype for computation (default: torch.float64).
    device : torch.device or str, optional
        Target device for computation (default: 'cpu').

    Returns
    -------
    torch.Tensor
        - If scalars are given → returns a single scalar tensor.
        - If vectors are given → returns a (n, m) matrix of spherical kernel values.
    """
    pi = torch.tensor(torch.pi, dtype=dtype, device=device)
    mia = pi / 180.0

    if lat2 is None or lon2 is None:
        lat2, lon2 = lat1, lon1

    lat1 = lat1.view(-1, 1)
    lon1 = lon1.view(-1, 1)
    lat2 = lat2.view(1, -1)
    lon2 = lon2.view(1, -1)

    a = torch.sin(lat1 * mia) * torch.sin(lat2 * mia) + torch.cos(lat1 * mia) * torch.cos(lat2 * mia) * torch.cos((lon1 - lon2) * mia)
    a = torch.clamp(a, -1.0, 1.0)
    theta = torch.acos(a)

    mask = torch.isclose(torch.cos(theta), torch.tensor(-1.0, dtype=dtype, device=device))
    upper = 0.5 + torch.cos(theta) / 2.0
    lower = 0.0
    num_steps = int(1e5)
    x_base = torch.linspace(0.0 + 1e-10, 1.0 - 1e-10, num_steps, dtype=dtype, device=device)
    x_vals = lower + (upper - lower).unsqueeze(0) * x_base.view(-1, 1, 1)
    y_vals = torch.log(1 - x_vals) / x_vals
    res_integral = torch.trapz(y_vals, x_base, dim=0)

    res = 1.0 - pi ** 2 / 6.0 - res_integral
    res[mask] = 1.0 - pi ** 2 / 6.0

    if res.numel() == 1:
        return res.squeeze()
    return res

def compute_spherical_kernel(
    KK: int,
    X: torch.Tensor,
    ggrids: torch.Tensor,
    Konev: torch.Tensor,
    eiKvecmval: torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    """
    Compute the multi-resolution thin-plate spline (TPS) basis matrix for points on a sphere.

    This function calculates the TPS basis for a set of target locations (`ggrids`) 
    relative to reference locations (`X`) using a spherical kernel based on great-circle 
    distances. It vectorizes the kernel computation and then projects residuals onto 
    precomputed eigenvectors to construct the final basis matrix.

    Parameters
    ----------
    KK : int
        Number of TPS basis functions.
    X : torch.Tensor
        Reference locations, shape (n, 2), where each row is (latitude, longitude).
    ggrids : torch.Tensor
        Target locations, shape (N, 2), where each row is (latitude, longitude).
    Konev : torch.Tensor
        Precomputed vector K @ onev, shape (n,), used for centering the kernel.
    eiKvecmval : torch.Tensor
        Eigenvectors divided by eigenvalues, shape (n, KK-1), used to construct higher-order TPS basis.
    dtype : torch.dtype, optional
        Desired precision (default: torch.float64).
    device : torch.device or str, optional
        Device for computation (default: 'cpu').

    Returns
    -------
    torch.Tensor
        TPS basis matrix of shape (N, KK), where each row corresponds to a target location 
        in `ggrids` and each column represents a TPS basis function. The first column is 
        the constant basis (normalized), and remaining columns are higher-order TPS functions.
    """
    N, n = ggrids.shape[0], X.shape[0]
    lat1, lon1 = ggrids[:, 0], ggrids[:, 1]
    lat2, lon2 = X[:, 0], X[:, 1]

    f2_matrix = spherical_kernel(lat1   = lat1,
                                 lon1   = lon1,
                                 lat2   = lat2,
                                 lon2   = lon2,
                                 dtype  = dtype,
                                 device = device
                                 )
    
    t_matrix = f2_matrix - Konev.view(1, -1)
    xx = torch.zeros((N, KK), dtype=dtype, device=device)
    xx[:, 0] = torch.sqrt(torch.tensor(1.0 / n, dtype=dtype, device=device))
    xx[:, 1:] = t_matrix @ eiKvecmval[:, :(KK - 1)]

    return xx

def mrts_sphere(
    knot: torch.Tensor,
    k: int,
    X: torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device: Union[str, torch.device] = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Compute multi-resolution thin-plate spline (MRTS) basis functions on spherical coordinates.

    This function constructs a TPS basis matrix for a set of target locations `X` 
    based on reference nodes `knot`. It uses a spherical kernel (great-circle distance)
    and projects the centered kernel onto precomputed eigenvectors to form higher-order
    basis functions. The first column is a normalized constant basis, and remaining
    columns are higher-order TPS functions.

    Parameters
    ----------
    knot : torch.Tensor
        Reference nodes, shape (n, 2), each row as (latitude, longitude in degrees).
    k : int
        Number of TPS basis functions to compute.
    X : torch.Tensor
        Target locations for evaluation, shape (N, 2), each row as (latitude, longitude).
    dtype : torch.dtype, optional
        Desired precision (default: torch.float64).
    device : str or torch.device, optional
        Device for computation (default: 'cpu').

    Returns
    -------
    dict
        Contains key "MRTS" mapping to the TPS basis matrix of shape (N, k).
        Each row corresponds to a target location in `X`.
    """
    n = knot.shape[0]
    onev = torch.linspace(1.0 / n, 1.0 / n, n, dtype=dtype, device=device).view(-1, 1)
    K = spherical_kernel(lat1   = knot[:, 0],
                         lon1   = knot[:, 1],
                         lat2   = knot[:, 0],
                         lon2   = knot[:, 1],
                         dtype  = dtype,
                         device = device
                         )
    Q = torch.eye(n, dtype=dtype, device=device) - (1.0 / n)
    eigenvalues, eigenvectors = torch.linalg.eigh(Q @ K @ Q)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx][:k]
    eigenvectors = eigenvectors[:, idx][:, :k]

    eiKvecmval = eigenvectors / eigenvalues.unsqueeze(0)

    dm_train = compute_spherical_kernel(KK          = k,
                                        X           = knot,
                                        ggrids      = X,
                                        Konev       = K @ onev,
                                        eiKvecmval  = eiKvecmval,
                                        dtype       = dtype,
                                        device      = device
                                        )

    return {"MRTS": dm_train}
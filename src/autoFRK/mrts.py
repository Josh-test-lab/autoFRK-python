"""
Title: Multi-Resolution Thin-plate Spline (MRTS) basis function for Spatial Data, and calculate the basis function by using rectangular or spherical coordinates
Author: Yao-Chih Hsu
Version: 1141025
Description: The MRTS method for autoFRK-Python project.
Reference: Resolution Adaptive Fixed Rank Kringing by ShengLi Tzeng & Hsin-Cheng Huang
"""

# import modules
import inspect
import numpy as np
import torch
import torch.nn as nn
from typing import Union, Dict, Optional
from .utils.logger import LOGGER, set_logger_level
from .utils.device import setup_device
from .utils.utils import to_tensor
from .utils.helper import subKnot, build_integral_table, integral_interpolator

global K_INT_TAB
K_INT_TAB = None

# function
# using in updateMrtsBasisComponents
# check = none
def create_rectangular_tps_matrix(
    s: torch.Tensor,
    tps_method: str = "rectangular",
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> torch.Tensor:
    """
    Construct the thin-plate spline (TPS) matrix for a set of spatial locations.

    This function computes the pairwise TPS values based on the distance matrix 
    derived from input locations. Optionally, distances can be computed using 
    spherical coordinates for global datasets.

    Parameters
    ----------
    s : torch.Tensor
        An (n, d) tensor representing the coordinates of n points in d-dimensional space.
    tps_method : str, optional
        Specifies the method used to compute thin-plate splines (TPS). Default is "rectangular".
        Options:
            - "rectangular": Compute TPS in Euclidean (rectangular) coordinates.
            - "spherical_fast": Use spherical coordinates but apply the rectangular TPS formulation for faster computation.
    dtype : torch.dtype, optional
        Data type of the output matrix. Default is torch.float64.
    device : torch.device or str, optional
        Device for computation. Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        An (n, n) symmetric thin-plate spline matrix.
    """
    d = s.shape[1]
    dist = calculate_distance(locs      = s,
                              new_locs  = s,
                              tps_method= tps_method
                              )
    
    if tps_method == "spherical_fast":
        d = 3

    L = tps_rectangular(dist   = dist,
                        d      = d,
                        dtype  = dtype,
                        device = device
                        )
    L = torch.triu(L, 1) + torch.triu(L, 1).T
    return L

# using in predictMrts
# check = none
def predict_rectangular_tps_matrix(
    s_new: torch.Tensor,
    s: torch.Tensor,
    tps_method: str = "rectangular",
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> torch.Tensor:
    """
    Compute the thin-plate spline (TPS) matrix between new locations and reference locations.

    The TPS matrix L is used in multi-resolution thin-plate spline basis computations.
    Each element L[i, j] represents the TPS kernel between the i-th row of `s_new` 
    and the j-th row of `s`.

    Parameters
    ----------
    s_new : torch.Tensor
        New locations at which TPS values are to be evaluated, shape (n1, d).
    s : torch.Tensor
        Reference locations corresponding to the TPS basis, shape (n2, d).
    tps_method : str, optional
        Specifies the method used to compute thin-plate splines (TPS). Default is "rectangular".
        Options:
            - "rectangular": Compute TPS in Euclidean (rectangular) coordinates.
            - "spherical_fast": Use spherical coordinates but apply the rectangular TPS formulation for faster computation.
    dtype : torch.dtype, optional
        Desired torch dtype for computation (default: torch.float64).
    device : torch.device or str, optional
        Target device for computation (default: 'cpu').

    Returns
    -------
    torch.Tensor, shape (n1, n2)
        TPS matrix, where element (i, j) is the thin-plate spline between 
        s_new[i] and s[j].
    """
    d = s.shape[1]
    dist = calculate_distance(locs      = s,
                              new_locs  = s_new,
                              tps_method= tps_method,
                              )
    
    if tps_method == "spherical_fast":
        d = 3

    L = tps_rectangular(dist   = dist,
                         d      = d,
                         dtype  = dtype,
                         device = device
                         )
            
    return L

# using in create_rectangular_tps_matrix, predict_rectangular_tps_matrix
# check = none
def tps_rectangular(
    dist: torch.Tensor,
    d: int,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> torch.Tensor:
    """
    Evaluate the thin-plate spline (TPS) radial basis function for given distances in rectangular coordinates.

    The TPS kernel depends on the dimension of the input points. This function
    supports 1D, 2D, and 3D points in rectangular (Euclidean) coordinates.

    Parameters
    ----------
    dist : torch.Tensor
        Pairwise distance matrix or vector.
    d : int
        Dimension of the positions (1, 2, or 3).
    dtype : torch.dtype, optional
        Data type of the output tensor. Default is torch.float64.
    device : torch.device or str, optional
        Device for computation. Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        TPS function evaluated at each element of `dist`. Shape matches `dist`.

    Raises
    ------
    ValueError
        If `d` is not 1, 2, or 3.

    Notes
    -----
    - 1D: TPS kernel is dist^3 / 12
    - 2D: TPS kernel is (dist^2 * log(dist)) / (8 * pi), with 0 handled separately
    - 3D: TPS kernel is -dist / 8
    """
    if d == 1:
        return dist ** 3 / 12
    elif d == 2:
        ret = torch.zeros_like(dist, dtype=dtype, device=device)
        mask = dist != 0
        ret[mask] = dist[mask]**2 * torch.log(dist[mask]) / (8 * torch.pi)
        return ret
    elif d == 3:
        return - dist / 8
    else:
        error_msg = f"Invalid dimension {d}, to calculate thin plate splines with rectangular coordinate, the dimension must be 1, 2, or 3."
        LOGGER.error(error_msg)
        raise ValueError(error_msg)

# using in create_rectangular_tps_matrix, predict_rectangular_tps_matrix
# check = none
def calculate_distance(
    locs: torch.Tensor,
    new_locs: Union[torch.Tensor, None],
    tps_method: str = "rectangular",
) -> torch.Tensor:
    """
    Compute pairwise distances between points, either in rectangular or spherical coordinates.

    Parameters
    ----------
    locs : torch.Tensor
        Tensor of shape (N, d) representing the coordinates of points. For spherical distances,
        columns are (latitude, longitude) in degrees.
    new_locs : torch.Tensor or None, optional
        Tensor of shape (M, d) for new locations to compute distances to. If None, computes
        distances among `locs` themselves. Default is None.
    tps_method : str, optional
        Specifies the method used to compute thin-plate splines (TPS). Default is "rectangular".
        Options:
            - "rectangular": Compute TPS in Euclidean (rectangular) coordinates.
            - "spherical_fast": Use spherical coordinates but apply the rectangular TPS formulation for faster computation.

    Returns
    -------
    torch.Tensor
        Pairwise distance matrix of shape (N, M) with distances between points.
        Distances are in the same units as coordinates (km for spherical).
    
    Notes
    -----
    - For rectangular coordinates, standard Euclidean distance is used.
    - For spherical coordinates, the great-circle distance formula is applied, assuming
      a sphere with radius 6371 km.
    - The function is vectorized for efficiency and supports GPU computation if `device`
      is specified.
    """
    if new_locs is None:
        new_locs = locs

    if tps_method == "rectangular":
        diff = new_locs[:, None, :] - locs[None, :, :]
        dist = torch.linalg.norm(diff, dim=2)

        return dist
    
    elif tps_method == "spherical_fast":
        if locs.ndim != 2 or new_locs.ndim != 2:
            error_msg = f"Invalid dimension of \"locs\" ({locs.ndim}) or \"new_locs\" ({new_locs.ndim}), to calculate thin plate splines with spherical coordinate, the dimension must be 2."
            LOGGER.error(error_msg)
            raise ValueError(error_msg)
        
        lat1 = locs[:, 0] * torch.pi / 180.0
        lon1 = locs[:, 1] * torch.pi / 180.0
        lat2 = new_locs[:, 0] * torch.pi / 180.0
        lon2 = new_locs[:, 1] * torch.pi / 180.0

        x1 = torch.cos(lat1) * torch.cos(lon1)
        y1 = torch.cos(lat1) * torch.sin(lon1)
        z1 = torch.sin(lat1)
        vec1 = torch.stack([x1, y1, z1], dim=1)

        x2 = torch.cos(lat2) * torch.cos(lon2)
        y2 = torch.cos(lat2) * torch.sin(lon2)
        z2 = torch.sin(lat2)
        vec2 = torch.stack([x2, y2, z2], dim=1)

        diff = vec1.unsqueeze(1) - vec2.unsqueeze(0)
        chord_len = torch.linalg.norm(diff, dim=2)

        radius = 1.0  # Earth's radius in kilometers is 6371.0
        dist = radius * 2 * torch.asin(torch.clamp(chord_len / 2, max=1.0))

        return dist
    
    else:
        error_msg = f'Invalid tps_method "{tps_method}", it should be one of ["rectangular", "spherical_fast"].'
        LOGGER.error(error_msg)
        ValueError(error_msg)

# using in MRTS.forward
# check = none
def compute_mrts_rectangular(
    s: torch.Tensor,
    xobs_diag: torch.Tensor,
    k: int,
    tps_method: str = "rectangular",
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> Dict[str, torch.Tensor]:
    """
    Compute core matrices for the Multi-Resolution Thin-Plate Spline (MRTS) method.

    This internal function is used in the MRTS forward pass to construct 
    the basis and projection matrices required for multi-resolution modeling.

    Parameters
    ----------
    s : torch.Tensor of shape (n, d)
        Position matrix of n locations in d dimensions.
    xobs_diag : torch.Tensor
        Observation matrix, typically diagonal or measurement values.
    k : int
        Number of eigenvalues/components to retain.
    tps_method : str, optional
        Specifies the method used to compute thin-plate splines (TPS). Default is "rectangular".
        Options:
            - "rectangular": Compute TPS in Euclidean (rectangular) coordinates.
            - "spherical_fast": Use spherical coordinates but apply the rectangular TPS formulation for faster computation.
    dtype : torch.dtype, optional
        Data type for computation (default: torch.float64).
    device : torch.device or str, optional
        Device for computation (default: 'cpu').

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing the core MRTS components:
        - **X** : torch.Tensor of shape (n, k)
            Base matrix for the first k components.
        - **UZ** : torch.Tensor of shape (n+d+1, k+d+1)
            Transformed matrix used for projection.
        - **BBBH** : torch.Tensor
            Projection matrix multiplied by Phi basis.
        - **nconst** : torch.Tensor
            Column normalization constants.
    """
    from .utils.predictor import updateMrtsBasisComponents, updateMrtsCoreComponentX, updateMrtsCoreComponentUZ

    # Update B, BBB, lambda, gamma
    Phi, B, BBB, lambda_, gamma = updateMrtsBasisComponents(s           = s,
                                                            k           = k,
                                                            tps_method  = tps_method,
                                                            dtype       = dtype,
                                                            device      = device
                                                            )
    
    # Update X, nconst
    X, nconst = updateMrtsCoreComponentX(s      = s,
                                         gamma  = gamma,
                                         k      = k,
                                         dtype  = dtype,
                                         device = device
                                         )

    # Update UZ
    UZ = updateMrtsCoreComponentUZ(s        = s,
                                   xobs_diag= xobs_diag,
                                   B        = B,
                                   BBB      = BBB,
                                   lambda_  = lambda_,
                                   gamma    = gamma,
                                   k        = k,
                                   dtype    =dtype,
                                   device   =device
                                   )

    return {
        "X":        X,
        "UZ":       UZ,
        "BBBH":     BBB @ Phi,
        "nconst":   nconst
    }

def compute_mrts_spherical(
    knot: torch.Tensor,
    k: int,
    X: torch.Tensor | None = None,
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
    if X is None:
        X = knot
    N, n = X.shape[0], knot.shape[0]
    K = tps_spherical(lat1   = knot[:, 0],
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
    Konev = K @ torch.linspace(1.0 / n, 1.0 / n, n, dtype=dtype, device=device).view(-1, 1)
    f2_matrix = tps_spherical(lat1   = X[:, 0],
                              lon1   = X[:, 1],
                              lat2   = knot[:, 0],
                              lon2   = knot[:, 1],
                              dtype  = dtype,
                              device = device
                              )
    
    t_matrix = f2_matrix - Konev.view(1, -1)
    ret = torch.zeros((N, k), dtype=dtype, device=device)
    ret[:, 0] = torch.sqrt(torch.tensor(1.0 / n, dtype=dtype, device=device))
    ret[:, 1:] = t_matrix @ eiKvecmval[:, :(k - 1)]

    return ret

def tps_spherical(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor | None = None,
    lon2: torch.Tensor | None = None,
    use_table: bool = False,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    """
    Compute thin-plate spline–like spherical kernel values between geographic coordinates.

    This function evaluates a smooth radial basis function (RBF) kernel adapted 
    for points on a sphere, based on great-circle (geodesic) distances between 
    two sets of latitude–longitude coordinates.

    It unifies the behavior of the legacy spherical kernel functions (`Kf` and `K`)
    by supporting both elementwise and full kernel matrix computations.

    The kernel is computed as:
        K(θ) = 1 - π² / 6 - ∫₀^{(1+cosθ)/2} [log(1 - t) / t] dt

    The integral term can be evaluated either numerically or via precomputed 
    interpolation using a lookup table for efficiency.

    Parameters
    ----------
    lat1 : torch.Tensor
        Tensor of latitudes (in degrees) for the first set of points, shape (n,).
    lon1 : torch.Tensor
        Tensor of longitudes (in degrees) for the first set of points, shape (n,).
    lat2 : torch.Tensor, optional
        Tensor of latitudes (in degrees) for the second set of points. 
        If None, `lat2 = lat1` is assumed.
    lon2 : torch.Tensor, optional
        Tensor of longitudes (in degrees) for the second set of points. 
        If None, `lon2 = lon1` is assumed.
    use_table : bool, optional
        Whether to use a precomputed numerical integration lookup table 
        (via :func:`build_integral_table`) for faster evaluation. 
        If False, the integral is computed directly using trapezoidal integration.
        Default is True.
    dtype : torch.dtype, optional
        Data type for computation (default: ``torch.float64``).
    device : torch.device or str, optional
        Target device for computation (default: ``'cpu'``).

    Returns
    -------
    torch.Tensor
        - If scalar coordinates are given → returns a single scalar kernel value.
        - If vector coordinates are given → returns a (n, m) kernel matrix.

    Notes
    -----
    - The kernel is symmetric and smooth, suitable for thin-plate spline (TPS)
      interpolation or Gaussian process models on spherical surfaces.
    - Internally uses great-circle distances computed from cosine law.
    - Numerical stability is ensured by clamping cosine values to [-1, 1].
    - Uses global cache ``K_INT_TAB`` when `use_table=True` for performance.

    See Also
    --------
    build_integral_table : Precompute lookup table for numerical integration.
    integral_interpolator : Interpolate definite integrals from a precomputed table.
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
    upper = torch.clamp(0.5 + torch.cos(theta) / 2.0, 0.0, 1.0)
    lower = 0.0
    num_steps = int(1e4)

    if use_table:
        global K_INT_TAB
        if K_INT_TAB is None:
            K_INT_TAB = build_integral_table(func       = lambda t: np.log1p(-t) / t,
                                             a          = 0.0,
                                             b          = 1.0,
                                             num_steps  = num_steps,
                                             eps        = 1e-12,
                                             dtype      = dtype,
                                             device     = device
                                             )
        res_integral = integral_interpolator(upper          = upper,
                                             lower          = lower,
                                             integral_table = K_INT_TAB,
                                             eps            = 1e-12
                                             )

    else:
        x_base = torch.linspace(0.0 + 1e-10, 1.0 - 1e-10, num_steps, dtype=dtype, device=device)
        x_vals = lower + (upper - lower).unsqueeze(0) * x_base.view(-1, 1, 1)
        y_vals = torch.log1p(- x_vals) / x_vals
        res_integral = torch.trapz(y_vals, x_base, dim=0)

    res = 1.0 - pi ** 2 / 6.0 - res_integral
    res[mask] = 1.0 - pi ** 2 / 6.0
    res = torch.clamp(res, max = 1.0)

    if res.numel() == 1:
        return res.squeeze()
    return res

# classes
class MRTS(nn.Module):
    """
    Multi-Resolution Thin-Plate Spline (MRTS) Basis Functions

    This class generates multi-resolution thin-plate spline basis functions, which are
    ordered by decreasing smoothness. Higher-order functions capture large-scale features,
    while lower-order functions capture small-scale details. These basis functions are
    typically used in spatio-temporal random effects models, such as Fixed Rank Kriging.

    Methods
    -------
    __init__(dtype=torch.float64, device='cpu')
        Initialize an MRTS object with specified dtype and computation device.

    forward(knot, k, x=None, maxknot=5000, tps_method="rectangular", dtype=torch.float64, device='cpu')
        Compute multi-resolution TPS basis functions at the given knot locations
        and optionally evaluate them at new locations.
    """
    def __init__(
        self,
        logger_level: int | str= 20,
        dtype: torch.dtype=torch.float64,
        device: Optional[Union[torch.device, str]]=None
    ):
        """
        Initialize an MRTS object.

        Parameters
        ----------
        logger_level : int, str, optional
            Logging level for the process (e.g., logging.INFO or 20). Default is 20.
            Possible values:
            - `logging.NOTSET` or 0        : No specific level; inherits parent logger level
            - `logging.DEBUG`  or 10       : Detailed debugging information
            - `logging.INFO`   or 20       : General information about program execution
            - `logging.WARNING` or 30     : Warning messages, indicate potential issues
            - `logging.ERROR`  or 40       : Error messages, something went wrong
            - `logging.CRITICAL` or 50    : Severe errors, program may not continue
        dtype : torch.dtype, optional
            Tensor data type for computation. Default is torch.float64.
        device : torch.device or str, optional
            Target device for computation ("cpu" or "cuda"). Default is "cpu".
            
        Raises
        ------
        TypeError
            If `dtype` is not a valid torch.dtype instance.
        """
        super().__init__()

        # set logger level
        if logger_level != 20:
            set_logger_level(LOGGER, logger_level)

        # setup device
        self.device = device

        # dtype check
        if not isinstance(dtype, torch.dtype):
            error_msg = f"Invalid dtype: expected a torch.dtype instance, got {type(dtype).__name__}"
            LOGGER.error(error_msg)
            raise TypeError(error_msg)
        self.dtype = dtype

    def forward(
        self,
        knot: torch.Tensor, 
        k: int=None, 
        x: torch.Tensor=None,
        maxknot: int=5000,
        tps_method: str | int="rectangular",
        dtype: torch.dtype=torch.float64,
        device: Optional[Union[torch.device, str]]=None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Multi-Resolution Thin-Plate Spline (MRTS) basis functions.

        The basis functions are ordered by decreasing smoothness: higher-order functions
        capture large-scale features, lower-order functions capture small-scale details.
        Useful for spatio-temporal random effects modeling.

        Parameters
        ----------
        knot : torch.Tensor
            An (m, d) tensor of knot locations (d <= 3). Missing values are not allowed.
        k : int
            Number of basis functions to generate (k <= m).
        x : torch.Tensor, optional
            An (n, d) tensor of locations at which to evaluate basis functions.
            If None, the basis is evaluated at the knots.
        maxknot : int, optional
            Maximum number of knots to use. If less than m, a subset of knots is selected deterministically.
            Default is 5000.
        tps_method : str or int, optional
            Specifies the method used to compute thin-plate splines (TPS). Default is "rectangular".
            Options:
                - "rectangular" (or 0): Compute TPS in Euclidean (rectangular) coordinates.
                - "spherical" (or 1): Compute TPS directly in spherical coordinates.
                - "spherical_fast" (or 2): Use spherical coordinates but apply the rectangular TPS formulation for faster computation.
        dtype : torch.dtype, optional
            Tensor data type for computation. Default is torch.float64.
        device : torch.device or str, optional
            Device for computation ("cpu" or "cuda"). Default is "cpu".

        Returns
        -------
        dict
            A dictionary containing:
            - **MRTS** : (n, k) tensor of basis function values at the evaluation locations
            - **UZ** : transformed matrix for internal computation (if available)
            - **Xu** : (n, d) tensor of unique knots used
            - **nconst** : normalization constants for each basis function
            - **BBBH** : (optional) projection matrix times Phi
            - **dtype** : data type used in computation
            - **device** : device used in computation
        """
        # logger it or not
        caller = inspect.stack()[1].frame.f_globals.get("__name__", "")
        use_logger = caller in ("__main__", "ipykernel_launcher")
        
        # setup device
        if device is None:
            device = setup_device(device = self.device,
                                  logger = use_logger
                                  )
            self.device = device
        else:
            # setup device
            device = setup_device(device = device,
                                  logger = use_logger
                                  )
            self.device = device

        # check dtype
        if dtype is None:
            dtype = self.dtype
        elif not isinstance(dtype, torch.dtype):
            warn_msg = f"Invalid dtype: expected a torch.dtype instance, got {type(dtype).__name__}, use default {self.dtype}"
            LOGGER.warning(warn_msg)
            dtype = self.dtype
        else:
            self.dtype = dtype

        # convert all major parameters
        xobs = to_tensor(obj   = knot,
                         dtype = dtype,
                         device= device
                         )
        x = to_tensor(obj   = x,
                      dtype = dtype,
                      device= device
                      )

        # check tps_method
        if not isinstance(tps_method, str):
            tps_method = int(tps_method)
        if tps_method == 0 or tps_method == "rectangular":
            tps_method = "rectangular"
            if use_logger:
                LOGGER.info(f'Calculate TPS with rectangular.')
        elif tps_method == 1 or tps_method == "spherical":
            tps_method = "spherical"
            if use_logger:
                LOGGER.info(f'Calculate TPS with spherical.')
        elif tps_method == 2 or tps_method == "spherical_fast":
            tps_method = "spherical_fast"
            if use_logger:
                LOGGER.info(f'Calculate TPS with spherical_fast.')
        else:
            warn_msg = f'Invalid tps_method "{tps_method}", it should be one of "rectangular", "spherical_fast", or "spherical", using default "rectangular" method instead.'
            LOGGER.warning(warn_msg)
        if tps_method == "spherical" and (xobs.ndim != 2 or xobs.shape[1] != 2):
            warn_msg = f'TPS method "spherical" requires the input locations "knot" to be a 2D matrix with shape (N, 2), but got shape {tuple(xobs.shape)}. Using default tps mothod "rectangular" instead.'
            LOGGER.warning(warn_msg)
            tps_method = "rectangular"
        self.tps_method = tps_method

        if xobs.ndim == 1:
            xobs = xobs.unsqueeze(1)
        Xu = torch.unique(xobs, dim=0)
        n, ndims = Xu.shape
        if x is None and n != xobs.shape[0]:
            x = xobs
        elif x is not None and x.ndim == 1:
            x = x.unsqueeze(1)
        
        if k < (ndims + 1):
            error_msg = f"k-1 can not be smaller than the number of dimensions!"
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

        if tps_method in ("rectangular", "spherical_fast"):
            if maxknot < n:
                Xu = subKnot(x      = Xu,
                             nknot  = maxknot,
                             xrng   = None, 
                             nsamp  = 1, 
                             dtype  = dtype,
                             device = device
                             )
                if x is None:
                    x = knot
                n = Xu.shape[0]

            xobs_diag = torch.diag(torch.sqrt(to_tensor(float(n) / float(n - 1), dtype=dtype, device=device)) / torch.std(xobs, dim=0, unbiased=True))
            
            if x is not None:
                if k - ndims - 1 > 0:
                    from .utils.predictor import predict_mrts_rectangular
                    result = predict_mrts_rectangular(s         = Xu,
                                                      xobs_diag = xobs_diag,
                                                      s_new     = x,
                                                      k         = k - ndims - 1,
                                                      tps_method= self.tps_method,
                                                      dtype     = dtype,
                                                      device    = device
                                                      )
                else:
                    shift = Xu.mean(dim=0, keepdim=True)
                    X2 = Xu - shift
                    nconst = torch.sqrt(torch.sum(X2**2, dim=0, keepdim=True))
                    X2 = torch.cat(
                        [
                            torch.ones((x.shape[0], 1), dtype=dtype, device=device),
                            ((x - shift) / nconst) * torch.sqrt(to_tensor(n, dtype=dtype, device=device))
                        ],
                        dim=1
                    )
                    result = {
                        "X": X2[:, :k]
                    }
                    x = None

            else:
                if k - ndims - 1 > 0:
                    result = compute_mrts_rectangular(s         = Xu,
                                                      xobs_diag = xobs_diag,
                                                      k         = k - ndims - 1,
                                                      tps_method= self.tps_method,
                                                      dtype     = dtype,
                                                      device    = device
                                                      )
                else:
                    shift = Xu.mean(dim=0, keepdim=True)
                    X2 = Xu - shift
                    nconst = torch.sqrt(torch.sum(X2**2, dim=0, keepdim=True))
                    X2 = torch.cat(
                        [
                            torch.ones((Xu.shape[0], 1), dtype=dtype, device=device),
                            ((Xu - shift) / nconst) * torch.sqrt(to_tensor(n, dtype=dtype, device=device))
                        ],
                        dim=1
                    )
                    result = {
                        "X": X2[:, :k]
                    }

            obj = {}
            obj["MRTS"] = result["X"]
            if result.get("nconst", None) is None:
                X2 = Xu - Xu.mean(dim=0, keepdim=True)
                result["nconst"] = torch.sqrt(torch.sum(X2**2, dim=0, keepdim=True))
            obj["UZ"] = result.get("UZ", None)
            obj["Xu"] = Xu
            obj["nconst"] = result.get("nconst", None)
            obj["BBBH"] = result.get("BBBH", None)
            
            obj["tps_method"] = self.tps_method
            obj["dtype"] = self.dtype
            obj["device"] = self.device

            if x is None:
                self.obj = obj
                return obj
            else:
                shift = Xu.mean(dim=0, keepdim=True)
                X2 = x - shift

                nconst = obj["nconst"]
                if nconst.dim() == 1:
                    nconst = nconst.unsqueeze(0)
                X2 = torch.cat(
                    [
                        torch.ones((X2.shape[0], 1), dtype=dtype, device=device),
                        X2 / nconst
                    ], 
                    dim=1
                )

                obj0 = obj
                if k - ndims - 1 > 0 and "X1" in result:
                    obj0["MRTS"] = torch.cat(
                        [
                            X2,
                            result.get("X1")
                        ],
                        dim=1
                    )
                else:
                    obj0["MRTS"] = X2

                self.obj = obj0
                return obj0
            
        elif tps_method == "spherical":
            res = compute_mrts_spherical(knot     = xobs,
                                         k        = k,
                                         X        = x,
                                         dtype    = dtype,
                                         device   = device,
                                         )
            obj = {}
            obj["MRTS"] = res
            obj["Xu"] = xobs
            obj["tps_method"] = self.tps_method
            obj["dtype"] = self.dtype
            obj["device"] = self.device
            
            return obj

    def predict(
        self,
        obj: Dict[str, torch.Tensor]=None,
        newx: Union[torch.Tensor, None] = None,
        tps_method: str | int | None = None,
        dtype: torch.dtype=torch.float64,
        device: Optional[Union[torch.device, str]]=None
    ) -> torch.Tensor:
        """
        Predict outputs using a trained MRTS (Multi-Resolution Thin-Plate Spline) model.

        Parameters
        ----------
        obj : dict of torch.Tensor, optional
            A dictionary containing model parameters and precomputed objects.
            If None, `self.obj` will be used (must have been set by a previous `forward` call).
            Keys commonly include:
                - 'M', 's', 'w', 'V', etc.
        newx : torch.Tensor, optional
            New input coordinates at which predictions are desired.
            If None, the method returns the internal object dictionary `obj` instead of predictions.
        tps_method : str, int or None, optional
            Specifies the method used to compute thin-plate splines (TPS). Default is None.
            Options:
                - None: Auto detect by `forward` method.
                - "rectangular" (or 0): Compute TPS in Euclidean (rectangular) coordinates.
                - "spherical" (or 1): Compute TPS directly in spherical coordinates.
                - "spherical_fast" (or 2): Use spherical coordinates but apply the rectangular TPS formulation for faster computation.
        dtype : torch.dtype, default=torch.float64
            The data type for computations. If different from the object's dtype, tensors will be converted.
        device : torch.device or str, optional
            The device on which computations will be performed (CPU or GPU). 
            If None, will use the device stored in `obj` or `self.device`.

        Returns
        -------
        torch.Tensor
            Predicted values at `newx` based on the MRTS model. 
            If `newx` is None, returns the internal object dictionary `obj`.

        Raises
        ------
        ValueError
            If neither `obj` is provided nor `self.obj` exists (i.e., `forward` has not been called).

        Notes
        -----
        - The method automatically handles conversion of tensor types and device placement.
        - Logs warnings when default values are used or when parameters have incompatible types.
        - Calls `predict_MRTS` from `autoFRK.utils.predictor` to perform the actual prediction computation.
        """
        if obj is None and not hasattr(self, "obj"):
            error_msg = f'No input "obj" is provided and `MRTS.forward` has not been called before `MRTS.predict`.'
            LOGGER.error(error_msg)
            raise ValueError(error_msg)
        elif obj is None and hasattr(self, "obj"):
            obj = self.obj

        # setup object type
        change_tensor = False

        # setup device
        obj['device'] = obj.get('device', None)
        if device is None:
            if obj['device'] is not None:
                device = obj['device']
            else:
                device = self.device
        elif device == obj['device']:
            device = obj['device']
        elif device == self.device:
            device = self.device
        else:
            caller = inspect.stack()[1].frame.f_globals.get("__name__", "")
            use_logger = caller in ("__main__", "ipykernel_launcher")
            device = setup_device(device = device,
                                  logger = use_logger
                                  )
            change_tensor = True
        self.device = device

        # check dtype
        obj['dtype'] = obj.get('dtype', None)
        if dtype is None:
            if obj['dtype'] is not None:
                dtype = obj['dtype']
            else:
                dtype = self.dtype
        elif dtype == obj['dtype']:
            dtype = obj['dtype']
        elif dtype == self.dtype:
            dtype = self.dtype
        elif not isinstance(dtype, torch.dtype):
            warn_msg = f"Invalid dtype: expected a torch.dtype instance, got {type(dtype).__name__}, use default {self.dtype}"
            LOGGER.warning(warn_msg)
            dtype = obj['dtype']
        else:
            change_tensor = True
        self.dtype = dtype

        # convert all major parameters
        if change_tensor:
            obj = to_tensor(obj     = obj,
                            dtype   = self.dtype,
                            device  = self.device
                            )
        
        if newx is None and obj is not None:
            return obj

        # check tps_method
        if tps_method is None:
            if obj.get('tps_method', None) is not None:
                tps_method = obj['tps_method']
            elif hasattr(self, "tps_method"):
                warn_msg = f'No input "tps_method" is provided, use the default value `"rectangular"` for MRTS.predict.'
                LOGGER.warning(warn_msg)
                tps_method = "rectangular"
            else:
                error_msg = f'Could not find the parameter "tps_method". Please specify a valid method ("rectangular", "spherical_fast" or "spherical").'
                LOGGER.error(error_msg)
                ValueError(error_msg)
        if not isinstance(tps_method, str):
            tps_method = int(tps_method)
        if tps_method == 0 or tps_method == "rectangular":
            tps_method = "rectangular"
        elif tps_method == 1 or tps_method == "spherical":
            tps_method = "spherical"
        elif tps_method == 2 or tps_method == "spherical_fast":
            tps_method = "spherical_fast"
        else:
            error_msg = f'Invalid tps_method "{tps_method}", it should be one of "rectangular", "spherical_fast", or "spherical".'
            LOGGER.error(error_msg)
            ValueError(error_msg)
        self.tps_method = tps_method

        from autoFRK.utils.predictor import predict_MRTS
        return predict_MRTS(obj         = obj,
                            newx        = newx,
                            tps_method  = self.tps_method,
                            dtype       = self.dtype,
                            device      = self.device
                            )

# main program
if __name__ == "__main__":
    print("This is the class `MRTS` for autoFRK package. Please import it in your code to use its functionalities.")









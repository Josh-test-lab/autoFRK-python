"""
Title: Multi-Resolution Thin-plate Spline (MRTS) basis function for Spatial Data, and calculate the basis function by using rectangular or spherical coordinates
Author: Yao-Chih Hsu
Version: 1141019
Description: The MRTS method for autoFRK-Python project.
Reference: Resolution Adaptive Fixed Rank Kringing by ShengLi Tzeng & Hsin-Cheng Huang
"""

# import modules
import inspect
import torch
import torch.nn as nn
from typing import Union, Dict
from .utils.logger import LOGGER
from .utils.device import setup_device
from .utils.utils import to_tensor
from .utils.helper import subKnot

# function
# using in updateMrtsBasisComponents
# check = none
def createThinPlateMatrix(
    s: torch.Tensor,
    calculate_with_spherical: bool=False,
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
    calculate_with_spherical : bool, optional
        If True, computes distances on a sphere (useful for latitude/longitude data). Default is False.
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
    dist = calculate_distance(locs                      = s,
                              new_locs                  = s,
                              calculate_with_spherical  = calculate_with_spherical,
                              )
    L = thinPlateSplines(dist   = dist,
                         d      = d,
                         dtype  = dtype,
                         device = device
                         )
    L = torch.triu(L, 1) + torch.triu(L, 1).T
    return L

# using in predictMrts
# check = none
def predictThinPlateMatrix(
    s_new: torch.Tensor,
    s: torch.Tensor,
    calculate_with_spherical: bool=False,
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
    calculate_with_spherical : bool, optional
        If True, distances are computed on the sphere instead of Euclidean.
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
    dist = calculate_distance(locs                      = s,
                              new_locs                  = s_new,
                              calculate_with_spherical  = calculate_with_spherical,
                              )
    L = thinPlateSplines(dist   = dist,
                         d      = d,
                         dtype  = dtype,
                         device = device
                         )
            
    return L

# using in createThinPlateMatrix, predictThinPlateMatrix
# check = none
def thinPlateSplines(
    dist: torch.Tensor,
    d: int,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> torch.Tensor:
    """
    Evaluate the thin-plate spline (TPS) radial basis function for given distances.

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

# using in createThinPlateMatrix, predictThinPlateMatrix
# check = none
def calculate_distance(
    locs: torch.Tensor,
    new_locs: Union[torch.Tensor, None],
    calculate_with_spherical: bool=False
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
    calculate_with_spherical : bool, optional
        If True, compute distances on the sphere (great-circle distance). Otherwise, use
        rectangular Euclidean distance. Default is False.

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

    if not calculate_with_spherical:
        diff = new_locs[:, None, :] - locs[None, :, :]
        dist = torch.linalg.norm(diff, dim=2)

        return dist
    
    else:
        if locs.ndim != 2 or new_locs.ndim != 2:
            error_msg = f"Invalid dimension of \"locs\" ({locs.ndim}) or \"new_locs\" ({new_locs.ndim}), to calculate thin plate splines with spherical coordinate, the dimension must be 2."
            LOGGER.error(error_msg)
            raise ValueError(error_msg)
        
        warn_msg = f"Calculating distances using spherical coordinates are experimental feature, please make sure your data are in latitude and longitude format (in degrees)."
        LOGGER.warning(warn_msg)
        
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

# using in MRTS.forward
# check = none
def computeMrts(
    s: torch.Tensor,
    xobs_diag: torch.Tensor,
    k: int,
    calculate_with_spherical: bool = False,
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
    calculate_with_spherical : bool, optional
        Whether to compute TPS distances using spherical coordinates. Default is False.
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
    Phi, B, BBB, lambda_, gamma = updateMrtsBasisComponents(s                       = s,
                                                            k                       = k,
                                                            calculate_with_spherical= calculate_with_spherical,
                                                            dtype                   = dtype,
                                                            device                  = device
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

    forward(knot, k, x=None, maxknot=5000, calculate_with_spherical=False, dtype=torch.float64, device='cpu')
        Compute multi-resolution TPS basis functions at the given knot locations
        and optionally evaluate them at new locations.
    """
    def __init__(
        self,
        dtype: torch.dtype=torch.float64,
        device: Union[torch.device, str]='cpu'
    ):
        """
        Initialize an MRTS object.

        Parameters
        ----------
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

        # setup device
        caller = inspect.stack()[1].frame.f_globals.get("__name__", "")
        use_logger = caller in ("__main__", "ipykernel_launcher")
        self.device = setup_device(device = device,
                                   logger = use_logger
                                   )

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
        calculate_with_spherical: bool = False,
        dtype: torch.dtype=torch.float64,
        device: Union[torch.device, str]='cpu'
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
        calculate_with_spherical : bool, optional
            If True, calculates TPS distances using spherical coordinates (for global data). Default is False.
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
        """
        # setup device
        if device is None:
            device = self.device
        else:
            # setup device
            caller = inspect.stack()[1].frame.f_globals.get("__name__", "")
            use_logger = caller in ("__main__", "ipykernel_launcher")
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

        # check calculate_with_spherical
        if type(calculate_with_spherical) is not bool:
            calculate_with_spherical = False
            LOGGER.warning(f'Parameter "calculate_with_spherical" should be a boolean, the type you input is "{type(calculate_with_spherical).__name__}". Default value \"False\" is used.')

        if not calculate_with_spherical:
            LOGGER.info(f'Calculate TPS with rectangular coordinates.')
        else:
            LOGGER.info(f'Calculate TPS with spherical coordinates.')
        self.calculate_with_spherical = calculate_with_spherical

        # convert all major parameters
        xobs = to_tensor(obj   = knot,
                         dtype = dtype,
                         device= device
                         )
        x = to_tensor(obj   = x,
                      dtype = dtype,
                      device= device
                      )
        
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

        if maxknot < n:
            bmax = maxknot
            Xu = subKnot(x      = Xu,
                         nknot  = bmax,
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
                from .utils.predictor import predictMrts
                result = predictMrts(s                          = Xu,
                                     xobs_diag                  = xobs_diag,
                                     s_new                      = x,
                                     k                          = k - ndims - 1,
                                     calculate_with_spherical   = calculate_with_spherical,
                                     dtype                      = dtype,
                                     device                     = device
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
                result = computeMrts(s                          = Xu,
                                     xobs_diag                  = xobs_diag,
                                     k                          = k - ndims - 1,
                                     calculate_with_spherical   = calculate_with_spherical,
                                     dtype                      = dtype,
                                     device                     = device
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
    def predict(
        self,
        obj: Dict[str, torch.Tensor]=None,
        newx: Union[torch.Tensor, None] = None,
        calculate_with_spherical: Union[bool, None] = None
    ) -> torch.Tensor:
        """

        """
        if obj is None and not hasattr(self, "obj"):
            error_msg = f'No input "obj" is provided and `MRTS.forward` has not been called before `MRTS.predict`.'
            LOGGER.error(error_msg)
            raise ValueError(error_msg)
        elif obj is None and hasattr(self, "obj"):
            obj = self.obj
        
        if newx is None and obj is not None:
            return obj
        
        if calculate_with_spherical is None and hasattr(self, "calculate_with_spherical"):
            warn_msg = f'No input "calculate_with_spherical" is provided, use the default value `False` for MMRTS.predict.'
            LOGGER.warning(warn_msg)
            calculate_with_spherical = False
        else:
            if calculate_with_spherical is None:
                calculate_with_spherical = self.calculate_with_spherical
            if type(calculate_with_spherical) is not bool:
                calculate_with_spherical = False
                LOGGER.warning(f'Parameter "calculate_with_spherical" should be a boolean, the type you input is "{type(calculate_with_spherical).__name__}". Default value \"False\" is used.')
            if not calculate_with_spherical:
                LOGGER.info(f'Calculate TPS with rectangular coordinates.')
            else:
                LOGGER.info(f'Calculate TPS with spherical coordinates.')

        from autoFRK.utils.predictor import predict_mrts
        return predict_mrts(obj                     = obj,
                            newx                    = newx,
                            calculate_with_spherical= calculate_with_spherical,
                            dtype                   = self.dtype,
                            device                  = self.device
                            )

# main program
if __name__ == "__main__":
    print("This is the class `MRTS` for autoFRK package. Please import it in your code to use its functionalities.")









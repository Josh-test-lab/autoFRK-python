"""
Title: Multi-Resolution Thin-plate Spline basis function for Spatial Data, and calculate the basis function by using rectangular or spherical coordinates.
Author: Hsu, Yao-Chih
Version: 1141017
Reviewer: 
Reviewed Version:
Description: 
Reference: Resolution Adaptive Fixed Rank Kringing by ShengLi Tzeng & Hsin-Cheng Huang
"""

# import modules
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import quad
from typing import Optional, Union, Any
from .utils.logger import setup_logger
from .utils.device import setup_device
from .utils.utils import *

# logger config
LOGGER = setup_logger()

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
    
    """
    d = s.shape[1]
    diff = s[:, None, :] - s[None, :, :]
    dist = torch.linalg.norm(diff, dim=2)
    L = thinPlateSplines(dist                       = dist,
                         calculate_with_spherical   = calculate_with_spherical,
                         d                          = d,
                         n_integral                 = None,
                         n_min                      = 1e4,
                         n_max                      = 1e10,
                         dtype                      = dtype,
                         device                     = device
                         )
    L = torch.triu(L, 1) + torch.triu(L, 1).T
    return L

# using in createThinPlateMatrix
# check = none
def thinPlateSplines(
    dist: torch.Tensor,
    calculate_with_spherical: bool=False,
    d: int = None,
    n_integral: int = None,
    n_min: int = 1e4,
    n_max: int = 1e10,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> torch.Tensor:
    """

    """
    if type(calculate_with_spherical) is not bool:
        calculate_with_spherical = False
        LOGGER.warning(f'Parameter "calculate_with_spherical" should be a boolean, the type you input is "{type(calculate_with_spherical).__name__}". Default value \"False\" is used.')

    if not calculate_with_spherical:
        LOGGER.info(f'Calculate TPS with rectangular coordinates.')
        return tps_rectangular(dist     = dist,
                               d        = d,
                               dtype    = dtype,
                               device   = device
                               )
    else:
        LOGGER.info(f'Calculate TPS with spherical coordinates.')

        error_msg = f"The feature \"Calculate TPS with spherical coordinates\" is currently not available."
        LOGGER.error(error_msg)
        raise NotImplementedError(error_msg)
    
        return tps_spherical(locs       = None,
                             n_integral = None,
                             n_min      = 1e4,
                             n_max      = 1e10,
                             dtype      = dtype,
                             device     = device
                             )

# using in thinPlateSplines
# check = none
def tps_rectangular(
    dist: torch.Tensor,
    d: int,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> torch.Tensor:
    """

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

# using in thinPlateSplines
# check = none
def tps_spherical(
    locs: torch.Tensor,
    n_integral: int = None,
    n_min: int = 1e4,
    n_max: int = 1e10,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = 'cpu'
) -> torch.Tensor:
    """
    Compute the TPS radial basis matrix using spherical coordinates (vectorized).

    Args:
        locs: [N, 2] tensor of locations (latitude, longitude in degrees)
        n_integral: number of subintervals for Simpson's Rule
    Returns:
        [N, N] TPS radial basis matrix
    """
    if locs.ndim != 2:
        error_msg = f"Invalid dimension {locs.ndim}, to calculate thin plate splines with spherical coordinate, the dimension must be 2."
        LOGGER.error(error_msg)
        raise ValueError(error_msg)

    # convert to radians
    locs = locs * torch.pi / 180.0
    x = locs[:, 0].unsqueeze(1)
    y = locs[:, 1].unsqueeze(1)

    # compute cos(theta) for all pairs
    sin_x = torch.sin(x)
    cos_x = torch.cos(x)
    cos_theta = sin_x @ sin_x.T + (cos_x @ cos_x.T) * torch.cos(y - y.T)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # initialize result
    ret = torch.ones_like(cos_theta, dtype=dtype, device=device) * (1 - (torch.pi ** 2) / 6)

    # mask for cos_theta != -1
    mask =  ~torch.eye(cos_theta.shape[0], dtype=torch.bool, device=cos_theta.device)
    upper = (cos_theta[mask] + 1) / 2  # upper limit of integral
    a = torch.zeros_like(upper, dtype=dtype, device=device)
    b = upper
    n = n_integral if n_integral is not None else torch.clamp((10 * torch.max(b - a)).ceil().to(torch.int64), n_min, n_max)
    n = int(n)
    h = (b - a) / n  # step size
    
    # x values for Simpson's rule
    xs = a.unsqueeze(1) + h.unsqueeze(1) * torch.arange(0, n + 1, dtype=dtype, device=device).unsqueeze(0)
    # integrand f(x) = log(1-x)/x
    f = torch.log(1 - xs) / xs
    f[:, 0] = 0.0  # handle x=0 singularity

    # Simpson coefficients
    coeff = torch.ones(n + 1, dtype=dtype, device=device)
    coeff[1:-1:2] = 4
    coeff[2:-2:2] = 2
    integral = (h / 3) * torch.sum(f * coeff, dim=1)

    # subtract integral from baseline
    ret[mask] -= integral
    ret[~mask] = 1.0

    return ret

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
    Compute MRTS (Multi-Resolution Thin-Plate Spline) core matrices.

    Parameters
    ----------
    s : (n, d) torch.Tensor
        Position matrix
    xobs_diag : torch.Tensor
        Observation matrix (typically diagonal or measurements)
    k : int
        Number of eigenvalues/components to keep
    dtype : torch.dtype
        Tensor dtype (default: torch.float64)
    device : torch.device or str
        Device (default: 'cpu')

    Returns
    -------
    dict containing:
        X     : (n, k) base matrix
        UZ    : (n+d+1, k+d+1) transformed matrix
        BBBH  : projection matrix * Phi
        nconst: column normalization constants
    """
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
    Multi-Resolution Thin-plate Spline Basis Functions
    """
    def __init__(
        self,
        dtype: torch.dtype=torch.float64,
        device: Union[torch.device, str]='cpu'
    ):
        """
        Initialize MRTS model.
        """
        super().__init__()

        # setup device
        self.device = setup_device(device=device)

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
    ) -> dict:
        """
        Forward pass to compute the basis functions. Constructs the basis functions based on the input locations and parameters.

        Returns:
            A tensor of shape [N, k] representing the basis functions.
        """
        # setup device
        if device is None:
            device = self.device
        else:
            device = setup_device(device=device)
            self.device = device

        # dtype check
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

            return obj0


# main program
if __name__ == "__main__":
    print("This is the class `MRTS` for autoFRK package. Please import it in your code to use its functionalities.")









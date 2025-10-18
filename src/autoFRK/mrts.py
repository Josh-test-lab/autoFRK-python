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









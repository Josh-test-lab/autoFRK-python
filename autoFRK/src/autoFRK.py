"""
Title: Automatic Fixed Rank Kriging.
Author: Hsu, Yao-Chih
Version: 1140727
Description: `autoFRK` is an R package to mitigate the intensive computation for modeling regularly/irregularly located spatial data using a class of basis functions with multi-resolution features and ordered in terms of their resolutions, and this project is to implement the `autoFRK` in Python.
Reference: Resolution Adaptive Fixed Rank Kringing by ShengLi Tzeng & Hsin-Cheng Huang
"""

# import modules
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, Union, Any, Dict
import datetime
from autoFRK.utils.logger import setup_logger
from autoFRK.utils.device import setup_device
from autoFRK.utils.utils import *
from autoFRK.utils.predictor import *

# logger config
LOGGER = setup_logger()

# class AutoFRK
class AutoFRK(nn.Module):
    """
    Automatic Fixed Rank Kriging

    This function performs resolution-adaptive fixed rank kriging on spatial
    data observed at one or multiple time points using a spatial random-effects
    model:

        z[t] = mu + G @ w[t] + eta[t] + e[t], 
        w[t] ~ N(0, M), 
        e[t] ~ N(0, s * D), 
        t = 1, ..., T

    where:
        - z[t] is an n-vector of (partially) observed data at n locations
        - mu is an n-vector of deterministic mean values
        - D is a given n x n covariance matrix of measurement errors
        - G is a given n x K basis function matrix
        - eta[t] is an n-vector of random variables corresponding to a stationary spatial process
        - w[t] is a K-vector of unobservable random weights

    Parameters:
        data (torch.Tensor): 
            n x T data matrix (can contain NaN) with z[t] as the t-th column.
        loc (torch.Tensor): 
            n x d matrix of coordinates corresponding to n locations.
        mu (torch.Tensor or float, optional): 
            n-vector or scalar for the mean mu. Default is 0.
        D (torch.Tensor, optional): 
            n x n covariance matrix for measurement errors. Default is identity matrix.
        G (torch.Tensor, optional): 
            n x K matrix of basis function values. Default is None (automatically determined).
        finescale (bool, optional): 
            If True, include an approximate stationary finer-scale process eta[t].
            Only the diagonals of D are used. Default is False.
        maxit (int, optional): 
            Maximum number of iterations. Default is 50.
        tolerance (float, optional): 
            Precision tolerance for convergence check. Default is 1e-6.
        maxK (int, optional): 
            Maximum number of basis functions considered. Default: 10*sqrt(n) if n>100 else n.
        Kseq (torch.Tensor, optional): 
            User-specified sequence of number of basis functions. Default is None.
        method (str, optional): 
            "fast" for k-nearest-neighbor imputation; "EM" for EM algorithm. Default is "fast".
        n_neighbor (int, optional): 
            Number of neighbors for the "fast" method. Default is 3.
        maxknot (int, optional): 
            Maximum number of knots for generating basis functions. Default is 5000.

    Returns:
        dict
            Object of class FRK, with keys:
                - 'M': ML estimate of M
                - 's': estimate of the scale parameter of measurement errors
                - 'negloglik': negative log-likelihood
                - 'w': K x T matrix with w[t] as the t-th column
                - 'V': K x K prediction error covariance matrix of w[t]
                - 'G': user-specified or automatically generated basis function matrix
                - 'LKobj': list from calling LKrig.MLE if useLK=True; else None

    Notes:
    Computes the ML estimate of M using the closed-form expression in
    Tzeng and Huang (2018). For large n, it is recommended to provide D as
    a sparse matrix.

    References:
    - Tzeng, S., & Huang, H.-C. (2018). Resolution Adaptive Fixed Rank Kriging,
      Technometrics. https://doi.org/10.1080/00401706.2017.1345701
    - Nychka D, Hammerling D, Sain S, Lenssen N (2016). LatticeKrig: Multiresolution
      Kriging Based on Markov Random Fields. R package version 8.4.
      https://github.com/NCAR/LatticeKrig
    """
    def __init__(
        self,
        mu: Union[float, torch.Tensor]=0.0, 
        D: torch.Tensor=None, 
        G: torch.Tensor=None,
        finescale: bool=False, 
        maxit: int=50, 
        tolerance: float=1e-6,
        maxK: int=None, 
        Kseq: torch.Tensor=None, 
        method: str="fast", 
        n_neighbor: int=3, 
        maxknot: int=5000,
        dtype: torch.dtype=torch.float64,
        device: Optional[Union[torch.device, str]]=None
        ):
        """
        Initialize autoFRK model with tensor-safe and device-aware configuration.
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

        # multi-GPU wrapper flag
        #self.dp_wrapper = torch.cuda.device_count() > 1

        # convert all major parameters
        self.mu = to_tensor(mu, dtype=dtype, device=self.device)
        self.D = to_tensor(D, dtype=dtype, device=self.device) if D is not None else None
        self.G = to_tensor(G, dtype=dtype, device=self.device) if G is not None else None
        self.Kseq = to_tensor(Kseq, dtype=dtype, device=self.device) if Kseq is not None else None

        # other parameters
        self.finescale = finescale
        self.maxit = maxit
        self.tolerance = tolerance
        self.maxK = maxK
        self.method = method
        self.n_neighbor = n_neighbor
        self.maxknot = maxknot

    def forward(
        self, 
        data: torch.Tensor, 
        loc: torch.Tensor
    ):
        """

        """
        data = to_tensor(data, dtype=self.dtype, device=self.device)
        loc = to_tensor(loc, dtype=self.dtype, device=self.device)
        
        data = data - self.mu
        if self.G is not None:
            Fk = self.G
        else:
            Fk = selectBasis(data           = data, 
                             loc            = loc,
                             D              = self.D, 
                             maxit          = self.maxit, 
                             avgtol         = self.tolerance,
                             max_rank       = self.maxK, 
                             sequence_rank  = self.Kseq, 
                             method         = self.method, 
                             num_neighbors  = self.n_neighbor,
                             max_knot       = self.maxknot, 
                             DfromLK        = None,
                             Fk             = None,
                             dtype          = self.dtype,
                             device         = self.device
                             )
        
        K = Fk.shape[1]
        if self.method == "fast":  # have OpenMP issue
            data = fast_mode_knn_sklearn(data       = data,
                                         loc        = loc, 
                                         n_neighbor = self.n_neighbor
                                         )
        elif self.method == "fast_faiss":
            data = fast_mode_knn_faiss(data         = data,
                                       loc          = loc, 
                                       n_neighbor   = self.n_neighbor
                                       )
        data = to_tensor(data, dtype=self.dtype, device=self.dtype)
        
        if not self.finescale:
            obj = indeMLE(data      = data,
                          Fk        = Fk[:, :K],
                          D         = self.D,
                          maxit     = self.maxit,
                          avgtol    = self.tolerance,
                          wSave     = True,
                          DfromLK   = None,
                          vfixed    = None,
                          verbose   = True,
                          dtype     = self.dtype,
                          device    = self.device
                          )
            
        else:
            """
            In the R package `autoFRK`, this functionality is implemented using the `LatticeKrig` package.
            This implementation is not provided in the current context.
            """
            error_msg = "The part about \"self.method == else\" in `AutoFRK.forward()` is Not provided yet!"
            LOGGER.error(error_msg)
            raise NotImplementedError(error_msg)

            # all codes here only for testing
            nu = 1
            nlevel = 3
            a_wght = None  # torch.Tensor or None
            NC = 10
            
            LK_obj = initializeLKnFRK(data=data,
                                      location=loc,
                                      nlevel=nlevel,
                                      weights=1.0 / torch.diag(self.D),
                                      n_neighbor=self.n_neighbor,
                                      nu=nu
                                      )
            
            DnLK = setLKnFRKOption(LK_obj,
                                   Fk[:, :K],
                                   nc=NC,
                                   a_wght=a_wght
                                   )
            DfromLK = DnLK['DfromLK']
            LKobj = DnLK['LKobj']
            obj = indeMLE(data=data,
                          Fk=Fk[:, :K],
                          D=self.D,
                          maxit=self.maxit,
                          avgtol=self.tolerance,
                          wSave=True,
                          DfromLK=DfromLK,
                          vfixed=DnLK.get('s', None)
                          )
        
        obj['G'] = Fk
        
        if self.finescale:
            """
            In the R package `autoFRK`, this functionality is implemented using the `LatticeKrig` package.
            This implementation is not provided in the current context.
            """
            error_msg = "The part about \"if self.finescale\" in `AutoFRK.forward()` is Not provided yet!"
            LOGGER.error(error_msg)
            raise NotImplementedError(error_msg)
        
            obj['LKobj'] = LKobj
            obj.setdefault('pinfo', {})
            obj['pinfo']["loc"] = loc
            obj['pinfo']["weights"] = 1.0 / torch.diag(self.D)
        else:
            obj['LKobj'] = None        
        
        self.obj = obj
        return obj
    
    def predict(
        self,
        object: dict = None,
        obsData: torch.Tensor = None,
        obsloc: torch.Tensor = None,
        mu_obs: Union[float, torch.Tensor] = 0,
        newloc: torch.Tensor = None,
        basis: torch.Tensor = None,
        mu_new: Union[float, torch.Tensor] = 0,
        se_report: bool = False
    ) -> dict:
        """
        
        """
        if object is None:
            object = self.obj
        return predict_FRK(object       = object,
                           obsData      = obsData,
                           obsloc       = obsloc,
                           mu_obs       = mu_obs,
                           newloc       = newloc,
                           basis        = basis,
                           mu_new       = mu_new,
                           se_report    = se_report,
                           dtype        = self.dtype,
                           device       = self.device
                           )


# main program
if __name__ == "__main__":
    print("This is the autoFRK package. Please import it in your code to use its functionalities.")

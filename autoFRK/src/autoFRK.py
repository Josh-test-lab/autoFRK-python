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
from typing import Optional, Union
import datetime
from autoFRK.utils.logger import setup_logger
from autoFRK.utils.device import setup_device
from autoFRK.utils.utils import *

# logger config
LOGGER = setup_logger()

# classes
class autoFRK(nn.Module):
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
    def __init__(self,
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
                 device: Optional[Union[torch.device, str]]=None
                 ):
        """
        
        """
        super().__init__()
        self.mu = mu
        self.D = D
        self.G = G
        self.finescale = finescale
        self.maxit = maxit
        self.tolerance = tolerance
        self.maxK = maxK
        self.Kseq = Kseq
        self.method = method
        self.n_neighbor = n_neighbor
        self.maxknot = maxknot
        self.device = setup_device(device=device)
        
        # 支援多GPU用 DataParallel
        if torch.cuda.device_count() > 1:
            self.dp_wrapper = True
        else:
            self.dp_wrapper = False

    def forward(self, 
                data: torch.Tensor, 
                loc: torch.Tensor
                ):
        """

        """
        data = data.to(self.device)
        loc = loc.to(self.device)
        
        # Step 1: 中心化
        data = data - self.mu
        
        # Step 2: 取得 basis Fk
        if self.G is not None:
            Fk = self.G.to(self.device)
        else:
            Fk = selectBasis(data, 
                              loc, 
                              self.D, 
                              self.maxit, 
                              self.tolerance,
                              self.maxK, 
                              self.Kseq, 
                              self.method, 
                              self.n_neighbor,
                              self.maxknot, 
                              None
                              ).to(self.device)
        
        K = Fk.shape[1]
        
        # Step 3: 若是 fast 方法，針對缺值用鄰近均值填補
        if self.method == "fast":
            data = fast_mode_knn(data=data,
                                 loc=loc, 
                                 n_neighbor=self.n_neighbor
                                 ).to(self.device)
        elif self.method == "fast_sklearn":  # avoid the OpenMP issue
            data = fast_mode_knn_sklearn(data=data,
                                         loc=loc, 
                                         n_neighbor=self.n_neighbor
                                         ).to(self.device)
        
        # Step 4: finescale 分支 or 否
        if not self.finescale:
            obj = indeMLE(data=data,
                          Fk=Fk[:, :K],
                          D=self.D.to(self.device) if self.D is not None else None,
                          maxit=self.maxit,
                          avgtol=self.tolerance,
                          wSave=True,
                          DfromLK=None)
        else:
            # 以下部分參數要根據你程式環境設定
            # 例如 nu, nlevel, a_wght, NC 等參數，你要自行定義或提供
            nu = 1
            nlevel = 3
            a_wght = None  # torch.Tensor or None
            NC = 10  # 假設一個值
            
            LK_obj = initializeLKnFRK(data=data, location=loc, nlevel=nlevel,
                                     weights=1.0 / torch.diag(self.D),
                                     n_neighbor=self.n_neighbor, nu=nu)
            
            DnLK = setLKnFRKOption(LK_obj, Fk[:, :K], nc=NC, a_wght=a_wght)
            DfromLK = DnLK['DfromLK']
            LKobj = DnLK['LKobj']
            
            # Depsilon = diag.spam(LK_obj$weight[LK_obj$pick], length(LK_obj$weight[LK_obj$pick])) 
            # 這段R特有，請自行處理
            
            obj = indeMLE(data=data,
                          Fk=Fk[:, :K],
                          D=self.D.to(self.device) if self.D is not None else None,
                          maxit=self.maxit,
                          avgtol=self.tolerance,
                          wSave=True,
                          DfromLK=DfromLK,
                          vfixed=DnLK.get('s', None))
            obj['LKobj'] = LKobj
        
        obj['G'] = Fk
        
        if not self.finescale:
            obj['LKobj'] = None
        
        # 返回一個 dict 結果（可依需求包成class）
        return obj


















# functions


# main program
if __name__ == "__main__":
    print("This is the autoFRK package. Please import it in your code to use its functionalities.")

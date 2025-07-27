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
            Fk = select_basis(data, 
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

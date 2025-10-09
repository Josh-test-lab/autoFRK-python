"""
Title: Predictor of autoFRK-Python Project
Author: Hsu, Yao-Chih
Version: 1141009
Reference:
"""

# development only
import os
import sys
sys.path.append(os.path.abspath("./src"))

# import modules
import os
import torch
import numpy as np
import faiss
import gc
from typing import Optional, Union
from autoFRK.utils.logger import setup_logger
from autoFRK.utils.utils import *
from autoFRK.mrts import MRTS

# logger config
LOGGER = setup_logger()

# predictor of autoFRK
# check = none
def predict_FRK(
    object: dict,
    obsData: torch.Tensor = None,
    obsloc: torch.Tensor = None,
    mu_obs: Union[float, torch.Tensor] = 0,
    newloc: torch.Tensor = None,
    basis: torch.Tensor = None,
    mu_new: Union[float, torch.Tensor] = 0,
    se_report: bool = False,
    device: Optional[Union[torch.device, str]] = 'cpu'
):
    """
    Predict method for Fixed Rank Kriging

    Predicted values and standard error estimates based on a model object 
    obtained from `autoFRK`.

    Parameters:
        object (dict): 
            A model object obtained from `autoFRK`.
        
        obsData (torch.Tensor, optional):
            Vector of observed data used for prediction.
            Default is None, which uses the `Data` field from `object`.
        
        obsloc (torch.Tensor, optional):
            Matrix with rows being coordinates of observation locations for `obsData`.
            Only models using `mrts` basis functions can have `obsloc` different from 
            the `loc` field in `object`. Not applicable for user-specified basis functions.
            Default is None, which uses the `loc` field from `object`.
        
        mu_obs (float or torch.Tensor, optional):
            Vector or scalar for the deterministic mean values at `obsloc`.
            Default is 0.
        
        newloc (torch.Tensor, optional):
            Matrix with rows being coordinates of new locations for prediction.
            Default is None, which gives prediction at the observed locations.
        
        basis (torch.Tensor, optional):
            Matrix where each column is a basis function evaluated at `newloc`.
            Can be omitted if the model was fitted using default `mrts` basis functions.
        
        mu_new (float or torch.Tensor, optional):
            Vector or scalar for the deterministic mean values at `newloc`.
            Default is 0.
        
        se_report (bool, optional):
            If True, the standard error of the prediction is also returned.
        
        device (torch.device or str, optional):
            Device to perform computation on. Default is `'cpu'`.
        
        dtype (torch.dtype, optional):
            Tensor dtype for computation. Default is `torch.float64`.

    Returns:
        dict
            A dictionary with the following components:
            
            - **pred_value** (`torch.Tensor`):  
            A matrix where element (i, t) is the predicted value at the i-th location and time t.
            
            - **se** (`torch.Tensor`, optional):  
            A vector where element i is the standard error of the predicted value at the i-th location.
            Only returned if `se_report=True`.
    """
    if basis is None:
        if newloc is None:
            if "G" not in object:
                error_msg = f"Basis matrix of new locations should be given (unless the model was fitted with mrts bases)!"
                LOGGER.error(error_msg)
                raise ValueError(error_msg)
            basis = object["G"]
        else:
            basis = predict_mrts(object["G"], newx=newloc, device=device)

    if basis.ndim == 1:
        basis = basis.unsqueeze(0)

    if obsloc is None:
        nobs = object["G"].shape[0]
    else:
        nobs = obsloc.shape[0]

    if obsData is not None:
        obsData = obsData - mu_obs
        if obsData.numel() != nobs:
            error_msg = f"Dimensions of obsloc and obsData are not compatible!"
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

    if newloc is not None:
        if basis.shape[0] != newloc.shape[0]:
            error_msg = f"Dimensions of obsloc and obsData are not compatible!"
            LOGGER.error(error_msg)
            raise ValueError(error_msg)
    else:
        if basis.shape[0] != object["G"].shape[0]:
            error_msg = f"Dimensions of obsloc and obsData are not compatible!"
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

    
    LKobj = object.get("LKobj", None)
    pinfo = object.get("pinfo", {})
    miss = object.get("missing", None)
    w = object["w"]
    
    if LKobj is None:
        if (obsloc is None) and (obsData is None):
            yhat = basis @ w

            if se_report:
                TT = w.shape[1] if w.ndim > 1 else 1
                if miss is None:
                    se_vec = torch.sqrt(torch.clamp(torch.sum((basis @ object["V"]) * basis, dim=1), min=0.0))
                    se = se_vec.unsqueeze(1).repeat(1, TT)
                else:
                    se = torch.full((basis.shape[0], TT), float('nan'), device=device)
                    pick = pinfo.get("pick", [])
                    D0 = pinfo["D"][pick][:, pick]
                    miss_bool = (miss["miss"] == 1).to(torch.bool)
                    Fk = object["G"][pick]
                    M = object["M"]
                    eigenvalues, eigenvectors = torch.linalg.eigh(M)
                    for tt in range(TT):
                        mask = ~miss_bool[:, tt]
                        if mask.sum().item() == 0:
                            continue
                        G = Fk[mask, :]
                        GM = G @ M
                        De = D0[mask][:, mask]
                        L = G @ eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0.0)))
                        V = M - GM.T @ invCz(object["s"] * De, L, GM)
                        se[:, tt] = torch.sqrt(torch.clamp(torch.sum((basis @ V) * basis, dim=1), min=0.0))

        if obsData is not None:
            pick = (~torch.isnan(obsData)).nonzero(asuple=True)[0].tolist()
            if obsloc is None:
                De = pinfo["D"][pick][:, pick]
                G = object["G"][pick]
            else:
                De = torch.eye(len(pick), device=device)
                G = predict_mrts(object["G"], newx=obsloc[pick])

            M = object["M"]
            GM = G @ M
            eigenvalues, eigenvectors = torch.linalg.eigh(M)
            L = G @ eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0.0)))
            yhat = basis @ GM.T @ invCz(object["s"] * De,
                                        L,
                                        obsData[pick]
                                        )

            if se_report:
                V = M - GM.T @ invCz(object["s"] * De,
                                     L,
                                     GM
                                     )
                se = torch.sqrt(torch.clamp(torch.sum(basis @ V * basis, dim=1), min=0.0)).unsqueeze(1)
                
    else:
        if obsData is None:
            if newloc is None:
                newloc = pinfo["loc"]
            info = LKobj["LKinfo.MLE"]
            phi0 = LKrig_basis(newloc,  # LKrig.basis is a function  # outside
                               info
                               )
            yhat = basis @ w + phi0 @ pinfo["wlk"]

            if se_report:
                TT = w.shape[1] if w.ndim > 1 else 1
                lambda_ = LKobj["lambda.MLE"] if isinstance(LKobj, dict) and "lambda.MLE" in LKobj else LKobj.get("lambda.MLE", None)
                loc = pinfo["loc"]
                pick = pinfo["pick"]
                G = object["G"][pick]
                M = object["M"]
                eigenvalues, eigenvectors = torch.linalg.eigh(M)
                L = G @ eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0.0)))
                phi1 = LKrig_basis(loc[pick],  # outside
                                   info
                                   )
                Q = LKrig_precision(info)  # outside
                weight = pinfo["weights"][pick]
                s = object["s"]
                phi0P = phi0 @ torch.linalg.inv(Q)
                if miss is None:
                    se_vec = LKpeon(M,
                                s,
                                G,
                                basis,
                                weight,
                                phi1,
                                phi0,
                                Q,
                                lambda_,
                                phi0P,
                                L,
                                only_se=True
                                )
                    se = se_vec.reshape(-1, TT)

                else:
                    se = torch.full((basis.shape[0], TT), float('nan'), device=device)
                    miss_bool = (miss["miss"] == 1).to(torch.bool)
                    for tt in range(TT):
                        mask = ~miss_bool[:, tt]
                        if mask.sum().item() == 0:
                            continue
                        se[:, tt] = LKpeon(M,
                                           s,
                                           G[mask, :],
                                           basis,
                                           weight[mask],
                                           phi1[mask, :],
                                           phi0,
                                           Q,
                                           lambda_,
                                           phi0P,
                                           L[mask, :],
                                           only_se=True
                                           )

        if obsData is not None:
            loc = pinfo["loc"]
            if newloc is None:
                newloc = loc
            pick = (~torch.isnan(obsData)).nonzero(asuple=True)[0].tolist()
            if obsloc is None:
                obsloc = loc
                De = pinfo["D"][pick][:, pick]
                G = object["G"][pick, :]
            else:
                G = predict_mrts(object["G"], newx=obsloc[pick, :])

            M = object["M"]
            eigenvalues, eigenvectors = torch.linalg.eigh(M)
            L = G @ eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0.0)))

            info = LKobj["LKinfo.MLE"]
            phi1 = LKrig_basis(obsloc[pick, :],  # outside
                               info
                               )
            Q = LKrig_precision(info)  # outside

            weight = torch.ones(len(pick), device=device)
            s = object["s"]
            phi0 = LKrig_basis(newloc, info)
            phi0P = phi0 @ torch.linalg.inv(Q)
            lambda_ = LKobj["lambda.MLE"] if isinstance(LKobj, dict) and "lambda.MLE" in LKobj else LKobj.get("lambda.MLE", None)

            pred = LKpeon(M,  # outside
                          s,
                          G,
                          basis,
                          weight,
                          phi1,
                          phi0,
                          Q,
                          lambda_,
                          phi0P,
                          L,
                          data=obsData[pick],
                          only_wlk=(not se_report)
                          )
            
            yhat = basis @ pred["w"] + phi0 @ pred["wlk"]
            if se_report:
                se = pred.get("se", None)

    if not se_report:
        return {"pred.value": (yhat + mu_new),
                "se": None
                }
    else:
        return {"pred.value": (yhat + mu_new),
                "se": se
                }

# predictor of autoFRK
# check = none
def predict_mrts(
    object: dict,
    newx: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = 'cpu'
) -> torch.Tensor:
    """
    Multi-Resolution Thin-plate Spline Basis Functions

    Evaluate multi-resolution thin-plate spline basis functions at given locations.
    This provides a generic prediction method for 'mrts' objects, 
    similar to `predict.ns` or `predict.bs` in R's 'splines' package.

    Parameters:
        object (dict): 
            Object produced from calling `mrts`. Must contain:
            - "Xu": torch.Tensor of knot locations
            - "nconst": normalization constants (1D tensor)
            - "BBBH": precomputed thin-plate spline matrix
            - "UZ": orthogonal basis
            - Itself (object["basis"]) representing the evaluated basis matrix.

        newx (torch.Tensor, optional): 
            (n × d) tensor of coordinates corresponding to n new locations.
            If None, returns `object` directly.

        device (torch.device or str, optional): 
            Device for computation (default 'cpu').

        dtype (torch.dtype, optional): 
            Tensor dtype for computation (default torch.float64).

    Returns:
        torch.Tensor
            (n × k) tensor of k basis function values at newx.
    """
    if newx is None:
        return object["basis"]

    Xu = object["Xu"].to(device=device)
    n = Xu.shape[0]
    xobs_diag = torch.diag(torch.sqrt(torch.tensor(n / (n - 1), device=device)) / Xu.std(dim=0, unbiased=True))
    ndims = Xu.shape[1]
    k = object["basis"].shape[1]
    x0 = newx.to(device=device)
    kstar = k - ndims - 1

    shift = Xu.mean(dim=0)
    nconst = object["nconst"].reshape(1, -1).to(device=device)
    X2 = torch.cat(
        [
            torch.ones((x0.shape[0], 1), device=device),
            (x0 - shift) / nconst
        ],
        dim=1
    )

    if kstar > 0:
        X1 = predictMrtsRcppWithBasis(Xu,
                                      xobs_diag,
                                      x0,
                                      object["BBBH"],
                                      object["UZ"],
                                      object["nconst"],
                                      k
                                      )["X1"]
        X1 = X1[:, :kstar]
        return torch.cat([X2, X1], dim=1)
    else:
        return X2





"""
Title: Predictor of autoFRK-Python Project
Author: Hsu, Yao-Chih
Version: 1141011
Reference:
"""

# import modules
import torch
from typing import Optional, Union, Dict
from autoFRK.utils.logger import setup_logger
from autoFRK.utils.device import *
from autoFRK.utils.utils import *

# logger config
LOGGER = setup_logger()

# predictor of autoFRK
# check = none
def predict_FRK(
    obj: dict,
    obsData: torch.Tensor = None,
    obsloc: torch.Tensor = None,
    mu_obs: Union[float, torch.Tensor] = 0,
    newloc: torch.Tensor = None,
    basis: torch.Tensor = None,
    mu_new: Union[float, torch.Tensor] = 0,
    se_report: bool = False,
    dtype: torch.dtype=torch.float64,
    device: Optional[Union[torch.device, str]] = 'cpu'
) -> dict:
    """
    Predict method for Fixed Rank Kriging

    Predicted values and standard error estimates based on a model object 
    obtained from `autoFRK`.

    Parameters:
        obj (dict): 
            A model object obtained from `autoFRK`.
        
        obsData (torch.Tensor, optional):
            Vector of observed data used for prediction.
            Default is None, which uses the `Data` field from `obj`.
        
        obsloc (torch.Tensor, optional):
            Matrix with rows being coordinates of observation locations for `obsData`.
            Only models using `mrts` basis functions can have `obsloc` different from 
            the `loc` field in `obj`. Not applicable for user-specified basis functions.
            Default is None, which uses the `loc` field from `obj`.
        
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
    obj = to_tensor(obj     = obj,
                    dtype   = dtype,
                    device  = device
                    )

    if basis is None:
        if newloc is None:
            if "G" not in obj:
                error_msg = f"Basis matrix of new locations should be given (unless the model was fitted with mrts bases)!"
                LOGGER.error(error_msg)
                raise ValueError(error_msg)
            basis = obj["G"]
            device = check_device(obj   = obj,
                                  device= device
                                  )

        else:
            newloc = to_tensor(obj      = newloc,
                               dtype    = dtype,
                               device   = device
                               )
            device = check_device(obj   = obj,
                                  device= device
                                  )
            basis = predict_mrts(obj    = obj["G"],
                                 newx   = newloc,
                                 dtype  = dtype,
                                 device = device
                                 )

    if basis.ndim == 1:
        basis = to_tensor(obj   = basis,
                          dtype = dtype,
                          device= device
                          )
        basis = basis.unsqueeze(0)

    if obsloc is None:
        nobs = obj["G"].shape[0]
    else:
        nobs = to_tensor(obj    = obsloc.shape[0],  
                         dtype  = dtype,
                         device = device
                         )

    if obsData is not None:
        obsData -= mu_obs
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
        if basis.shape[0] != obj["G"].shape[0]:
            error_msg = f"Dimensions of obsloc and obsData are not compatible!"
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

    
    LKobj = obj.get("LKobj", None)
    pinfo = obj.get("pinfo", {})
    miss = obj.get("missing", None)
    w = obj["w"]
    
    if LKobj is None:
        if (obsloc is None) and (obsData is None):
            yhat = basis @ w

            if se_report:
                TT = w.shape[1] if w.ndim > 1 else 1
                if miss is None:
                    se_vec = torch.sqrt(torch.clamp(torch.sum((basis @ obj["V"]) * basis, dim=1), min=0.0))
                    se = se_vec.unsqueeze(1).repeat(1, TT)
                else:
                    se = torch.full((basis.shape[0], TT), float('nan'), device=device)
                    pick = pinfo.get("pick", [])
                    D0 = pinfo["D"][pick][:, pick]
                    miss_bool = (miss["miss"] == 1).to(torch.bool)
                    Fk = obj["G"][pick]
                    M = obj["M"]
                    eigenvalues, eigenvectors = torch.linalg.eigh(M)
                    for tt in range(TT):
                        mask = ~miss_bool[:, tt]
                        if mask.sum().item() == 0:
                            continue
                        G = Fk[mask, :]
                        GM = G @ M
                        De = D0[mask][:, mask]
                        L = G @ eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0.0)))
                        V = M - GM.T @ invCz(obj["s"] * De, L, GM)
                        se[:, tt] = torch.sqrt(torch.clamp(torch.sum((basis @ V) * basis, dim=1), min=0.0))

        if obsData is not None:
            pick = (~torch.isnan(obsData)).nonzero(asuple=True)[0].tolist()
            if obsloc is None:
                De = pinfo["D"][pick][:, pick]
                G = obj["G"][pick]
            else:
                De = torch.eye(len(pick), dtype=dtype, device=device)
                G = predict_mrts(obj    = obj["G"],
                                 newx   = obsloc[pick],
                                 dtype  = dtype,
                                 device = device
                                 )

            M = obj["M"]
            GM = G @ M
            eigenvalues, eigenvectors = torch.linalg.eigh(M)
            L = G @ eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0.0)))
            yhat = basis @ GM.T @ invCz(R       = obj["s"] * De,
                                        L       = L,
                                        z       = obsData[pick],
                                        dtype   = dtype,
                                        device  = device
                                        )

            if se_report:
                V = M - GM.T @ invCz(R      = obj["s"] * De,
                                     L      = L,
                                     z      = GM,
                                     dtype  = dtype,
                                     device = device
                                     )
                se = torch.sqrt(torch.clamp(torch.sum(basis @ V * basis, dim=1), min=0.0)).unsqueeze(1)
                
    else:
        """
        In the R package `autoFRK`, this functionality is implemented using the `LatticeKrig` package.
        This implementation is not provided in the current context.
        """
        error_msg = "The part about \"LKobj is not None\" in `predict_FRK` is Not provided yet!"
        LOGGER.error(error_msg)
        raise NotImplementedError(error_msg)

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
                G = obj["G"][pick]
                M = obj["M"]
                eigenvalues, eigenvectors = torch.linalg.eigh(M)
                L = G @ eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0.0)))
                phi1 = LKrig_basis(loc[pick],  # outside
                                   info
                                   )
                Q = LKrig_precision(info)  # outside
                weight = pinfo["weights"][pick]
                s = obj["s"]
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
                G = obj["G"][pick, :]
            else:
                G = predict_mrts(obj["G"], newx=obsloc[pick, :])

            M = obj["M"]
            eigenvalues, eigenvectors = torch.linalg.eigh(M)
            L = G @ eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0.0)))

            info = LKobj["LKinfo.MLE"]
            phi1 = LKrig_basis(obsloc[pick, :],  # outside
                               info
                               )
            Q = LKrig_precision(info)  # outside

            weight = torch.ones(len(pick), device=device)
            s = obj["s"]
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
    obj: dict,
    newx: Optional[torch.Tensor] = None,
    dtype: torch.dtype=torch.float64,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Multi-Resolution Thin-plate Spline Basis Functions

    Evaluate multi-resolution thin-plate spline basis functions at given locations.
    This provides a generic prediction method for 'mrts' objects, 
    similar to `predict.ns` or `predict.bs` in R's 'splines' package.

    Parameters:
        obj (dict): 
            Object produced from calling `mrts`. Must contain:
            - "Xu": torch.Tensor of knot locations
            - "nconst": normalization constants (1D tensor)
            - "BBBH": precomputed thin-plate spline matrix
            - "UZ": orthogonal basis
            - Itself (obj["basis"]) representing the evaluated basis matrix.

        newx (torch.Tensor, optional): 
            (n × d) tensor of coordinates corresponding to n new locations.
            If None, returns `obj` directly.

        device (torch.device or str, optional): 
            Device for computation (default 'cpu').

        dtype (torch.dtype, optional): 
            Tensor dtype for computation (default torch.float64).

    Returns:
        torch.Tensor
            (n × k) tensor of k basis function values at newx.
    """
    if newx is None:
        return obj["basis"]

    Xu = obj["Xu"]
    n = Xu.shape[0]
    xobs_diag = torch.diag(torch.sqrt(torch.tensor(n / (n - 1), dtype=dtype, device=device)) / Xu.std(dim=0, unbiased=True))
    ndims = Xu.shape[1]
    k = obj["basis"].shape[1]
    x0 = newx
    kstar = k - ndims - 1

    shift = Xu.mean(dim=0)
    nconst = obj["nconst"].reshape(1, -1)
    X2 = torch.cat(
        [
            torch.ones((x0.shape[0], 1), dtype=dtype, device=device),
            (x0 - shift) / nconst
        ],
        dim=1
    )

    if kstar > 0:
        X1 = predictMrtsWithBasis(s         = Xu,
                                  xobs_diag = xobs_diag,
                                  s_new     = x0,
                                  BBBH      = obj["BBBH"],
                                  UZ        = obj["UZ"],
                                  nconst    = obj["nconst"],
                                  k         = k,
                                  dtype     = dtype,
                                  device    = device
                                  )["X1"]
        
        X1 = X1[:, :kstar]
        return torch.cat([X2, X1], dim=1)
    else:
        return X2

# using in predict_mrts
# check = none
def predictMrtsWithBasis(
    s: torch.Tensor,
    xobs_diag: torch.Tensor,
    s_new: torch.Tensor,
    BBBH: torch.Tensor,
    UZ: torch.Tensor,
    nconst: torch.Tensor,
    k: int,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> Dict[str, torch.Tensor]:
    """
    Internal function: Predict on new locations by MRTS method (PyTorch version)

    Parameters:
        s (torch.Tensor): 
            Location matrix, shape (n, d)
        xobs_diag (torch.Tensor): 
            Matrix of observations, shape (n, n)
        s_new (torch.Tensor): 
            New location matrix, shape (n2, d)
        BBBH (torch.Tensor): 
            Matrix for internal computing use, shape (d+1, k)
        UZ (torch.Tensor): 
            Matrix for internal computing use, shape (n, k)
        nconst (torch.Tensor): 
            Vector of column means, shape (d+1,)
        k (int): 
            Rank
        device (Union[torch.device, str], optional): 
            Device to use ("cpu" or "cuda"), if not consistent with inputs, 
            will be auto-detected using `check_device()`.

    Returns:
        Dict[str, torch.Tensor]
            {
                "X": s,
                "UZ": UZ,
                "BBBH": BBBH,
                "nconst": nconst,
                "X1": torch.Tensor
            }
    """
    n, d = s.shape
    n2 = s_new.shape[0]
    Phi_new = torch.zeros((n2, n), device=s.device)
    Phi_new = predictThinPlateMatrix(s_new  = s_new,
                                     s      = s,
                                     L      = Phi_new
                                     )

    X1 = Phi_new @ UZ[:, :k]
    B = torch.ones((n2, d + 1), dtype=dtype, device=device)
    B[:, -d:] = s_new
    X1_adjusted = X1 - B @ (BBBH @ UZ[:, :k].T)

    out = {
                "X": s,
                "UZ": UZ,
                "BBBH": BBBH,
                "nconst": nconst,
                "X1": X1_adjusted
            }
    return out

# using in predictMrtsWithBasis
# check = none
def predictThinPlateMatrix(
    s_new: torch.Tensor,
    s: torch.Tensor,
    L: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Thin Plate Spline (TPS) basis matrix between new and reference locations.
    
    Parameters:
        s_new (torch.Tensor): 
            New position matrix (n1 x d)
        s (torch.Tensor): 
            Reference position matrix (n2 x d)
        L (torch.Tensor): 
            Output matrix (n1 x n2), will be filled in-place

    Return:
    
    """
    dist = torch.norm(s_new.unsqueeze(1) - s.unsqueeze(0), dim=2)
    L[:] = thinPlateSplines(dist= dist,
                            d   = s_new.shape[1]
                            )

    return L

# using in predictThinPlateMatrix
# check = none
def thinPlateSplines(
    dist: torch.Tensor,
    d: int
) -> torch.Tensor:
    """
    Compute Thin Plate Splines (TPS) radial basis function.

    Parameters
    ----------
    dist : torch.Tensor
        Distance tensor (non-negative).
    d : int
        Dimension of the positions (1, 2, or 3).

    Returns
    -------
    torch.Tensor
        TPS value corresponding to each element in `dist`.
    """
    if d == 1:
        ret = dist.pow(3) / 12.0

    elif d == 2:
        ret = torch.zeros_like(dist)
        nonzero_mask = dist != 0
        ret[nonzero_mask] = ((dist[nonzero_mask] ** 2) * torch.log(dist[nonzero_mask])) / (8.0 * torch.pi)

    elif d == 3:
        ret = -dist / 8.0

    else:
        error_msg = f"Invalid dimension: d must be 1, 2, or 3."
        LOGGER.error(error_msg)
        raise ValueError(error_msg)

    return ret


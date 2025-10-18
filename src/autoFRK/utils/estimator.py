"""
Title: Setup file of autoFRK-Python Project
Author: Hsu, Yao-Chih
Version: 1141017
Reviewer: 
Reviewed Version:
Reference: 
Description: 
"""

# development only
#import os
#import sys
#sys.path.append(os.path.abspath("./src"))

# import modules
import torch
import numpy as np
import faiss
import gc
from typing import Optional, Dict, Union, Any
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import eigsh
from autoFRK.utils.logger import setup_logger

# logger config
LOGGER = setup_logger()

# compute negative log likelihood for autoFRK, using in selectBasis
# check = ok
def cMLE(
    Fk: torch.Tensor,
    num_columns: int,
    sample_covariance_trace: float,
    inverse_square_root_matrix: torch.Tensor,
    matrix_JSJ: torch.Tensor,
    s: float = 0,
    ldet: float = 0,
    wSave: bool = False,
    onlylogLike: bool = None,
    vfixed: float = None,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = 'cpu'
) -> dict:
    """
    Internal function: maximum likelihood estimate with the likelihood

    Parameters:
        Fk: (N, K) torch.Tensor, basis functions.
        num_columns: (int) Number of columns in the data.
        sample_covariance_trace: (float) Trace of the sample covariance matrix.
        inverse_square_root_matrix: (N, K) torch.Tensor, inverse square root matrix.
        matrix_JSJ: (K, K) torch.Tensor, covariance-like matrix.
        s: (float) Effective sample size, default is 0.
        ldet: (float) Log determinant of the transformation matrix, default is 0.
        wSave: (bool) Whether to save the L matrix, default is False.
        onlylogLike: (bool) If True, only return the negative log likelihood.
        vfixed: (float, optional) Fixed noise variance, if provided.
        device: (str) 'cpu' or 'cuda'.

    Returns:
        dict: {
            'v': (float) Estimated noise variance,
            'M': (torch.Tensor) Matrix M,
            's': (int) Effective sample size,
            'negloglik': (float) Negative log likelihood,
            'L': (torch.Tensor, optional) L matrix if wSave is True.
        }
    """
    nrow_Fk = Fk.shape[0]

    likelihood_object = computeNegativeLikelihood(
        nrow_Fk                 = nrow_Fk,
        ncol_Fk                 = Fk.shape[1],
        s                       = s,
        p                       = num_columns,
        matrix_JSJ              = matrix_JSJ,
        sample_covariance_trace = sample_covariance_trace,
        vfixed                  = vfixed,
        ldet                    = ldet,
        dtype                   = dtype,
        device                  = device
    )

    negative_log_likelihood = likelihood_object['negative_log_likelihood']

    if onlylogLike:
        return {'negloglik': negative_log_likelihood}

    P = likelihood_object['P']
    d_hat = likelihood_object['d_hat']
    v = likelihood_object['v']
    M = inverse_square_root_matrix @ P @ (torch.diag(d_hat) @ P.T) @ inverse_square_root_matrix

    if not wSave:
        L = None
    elif d_hat[0] != 0:
        L = Fk @ ((torch.diag(torch.sqrt(d_hat)) @ P.T) @ inverse_square_root_matrix)
        L = L[:, d_hat > 0]
    else:
        L = torch.zeros((nrow_Fk, 1), dtype=dtype, device=device)

    return {'v': v,
            'M': M,
            's': s,
            'negloglik': negative_log_likelihood,
            'L': L
            }

# independent maximum likelihood estimation for autoFRK, using in selectBasis
# check = ok
def indeMLE(
    data: torch.Tensor,
    Fk: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    maxit: int = 50,
    avgtol: float = 1e-6,
    wSave: bool = False,
    DfromLK: Optional[dict] = None,
    vfixed: Optional[float] = None,
    verbose: bool = True,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> dict:
    """
    Internal function: indeMLE

    Parameters:
        data (torch.Tensor): 
            n x T data matrix (NA allowed), where each column is z[t].
        Fk (torch.Tensor): 
            n x K matrix of basis function values; each column is a basis function evaluated at observation locations.
        D (torch.Tensor, optional): 
            n x n diagonal matrix.
        maxit (int, default 50): 
            Maximum number of iterations.
        avgtol (float, default 1e-6): 
            Average tolerance for convergence.
        wSave (bool, default False): 
            Whether to compute and return weight and covariance matrices.
        DfromLK (dict, optional): 
            n x n matrix or dictionary of low-rank kernel precomputations.
        vfixed (float, optional): 
            Fixed variance parameter (if provided, overrides estimation).
        verbose (bool, default True): 
            Print useful information during computation.
        device (str or torch.device, default 'cpu'): 
            Device to perform computation on.

    Returns:
        dict
            Dictionary containing estimated matrices, variance parameters, and optional diagnostic information.
    """
    withNA = torch.isnan(data).any().item()

    TT = data.shape[1]
    empty = torch.isnan(data).all(dim=0)
    notempty = (~empty).nonzero(as_tuple=True)[0]
    if empty.any():
        data = data[:, notempty]

    del_rows = torch.isnan(data).all(dim=1).nonzero(as_tuple=True)[0]
    pick = torch.arange(data.shape[0], dtype=torch.int64, device=device)

    if D is None:
        D = torch.eye(data.shape[0], dtype=dtype, device=device)

    if not isDiagonal(D):
        D0 = D
    else:
        D0 = torch.diag(torch.diag(D))

    if withNA and len(del_rows) > 0:
        pick = pick[~torch.isin(pick, del_rows)]
        data = data[~torch.isin(torch.arange(data.shape[0], dtype=torch.int64, device=device), del_rows), :]
        Fk = Fk[~torch.isin(torch.arange(Fk.shape[0], dtype=dtype, device=device), del_rows), :]
        if not torch.allclose(D, torch.diag(torch.diagonal(D))):
            D = D[~torch.isin(torch.arange(D.shape[0], dtype=dtype, device=device), del_rows)][:, ~torch.isin(torch.arange(D.shape[1], dtype=dtype, device=device), del_rows)]
        else:
            keep_mask = ~torch.isin(torch.arange(D.shape[0], dtype=torch.int64, device=device), del_rows)
            full_diag = torch.zeros(D.shape[0], dtype=dtype, device=device)
            full_diag[keep_mask] = torch.diagonal(D)[keep_mask]
            D = torch.diag(full_diag)
        withNA = torch.isnan(data).any().item()

    N = data.shape[0]
    K = Fk.shape[1]
    Depsilon = D
    is_diag = torch.allclose(D, torch.diag(torch.diagonal(D)))
    mean_diag = torch.mean(torch.diagonal(D))
    isimat = is_diag and torch.allclose(torch.diagonal(Depsilon), mean_diag.repeat(N), atol=1e-10)

    if not withNA:
        if isimat and DfromLK is None:
            sigma = 0  # we cannot find `.Option$sigma_FRK` in the R code  # outside
            out = cMLEimat(Fk           = Fk, 
                           data         = data, 
                           s            = sigma, 
                           wSave        = wSave,
                           S            = None,
                           onlylogLike  = None,
                           dtype        = dtype,
                           device       = device
                           )
            if out['v'] is not None:
                out['s'] = out['v'] if sigma == 0 else sigma
                out.pop("v", None)
            if wSave:
                w = torch.zeros((K, TT), dtype=dtype, device=device)
                w[:, notempty] = out['w']
                out['w'] = w
                out['pinfo'] = {'D': D0, 
                                'pick': pick
                                }
            return out
        
        elif DfromLK is None:
            out = cMLEsp(Fk         = Fk, 
                         data       = data, 
                         Depsilon   = Depsilon, 
                         wSave      = wSave,
                         dtype      = dtype,
                         device     = device
                         )
            if wSave:
                w = torch.zeros((K, TT), dtype=dtype, device=device)
                w[:, notempty] = out['w']
                out['w'] = w
                out['pinfo'] = {'D': D0, 
                                'pick': pick
                                }
            return out
        
        else:
            out = cMLElk(Fk         = Fk, 
                         data       = data, 
                         Depsilon   = Depsilon, 
                         wSave      = wSave, 
                         DfromLK    = DfromLK, 
                         vfixed     = vfixed,
                         dtype      = dtype,
                         device     = device
                         )
            if wSave:
                w = torch.zeros((K, TT), dtype=dtype, device=device)
                w[:, notempty] = out['w']
                out['w'] = w
            return out
        
    else:
        out = EM0miss(Fk        = Fk, 
                      data      = data, 
                      Depsilon  = Depsilon, 
                      maxit     = maxit, 
                      avgtol    = avgtol, 
                      wSave     = wSave,
                      DfromLK   = DfromLK, 
                      vfixed    = vfixed, 
                      verbose   = verbose,
                      dtype     = dtype,
                      device    = device
                      )
        if wSave:
            w = torch.zeros((K, TT), dtype=dtype, device=device)
            w[:, notempty] = out['w']
            out['w'] = w
            if DfromLK is None:
                out['pinfo'] = {'D': D0,
                                'pick': pick
                                }
        return out

# using in indeMLE
# check = ok
def cMLEimat(
    Fk: torch.Tensor,
    data: torch.Tensor,
    s: float,
    wSave: bool = False,
    S: Optional[torch.Tensor] = None,
    onlylogLike: Optional[bool] = None,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> dict:
    """
    
    """
    if onlylogLike is None:
        onlylogLike = not wSave

    num_columns = data.shape[1]
    nrow_Fk, ncol_Fk = Fk.shape

    projection = computeProjectionMatrix(Fk1    = Fk, 
                                         Fk2    = Fk, 
                                         data   = data, 
                                         S      = S, 
                                         dtype  = dtype, 
                                         device = device
                                         )
    inverse_square_root_matrix = projection["inverse_square_root_matrix"]
    matrix_JSJ = projection["matrix_JSJ"]

    sample_covariance_trace = torch.sum(data ** 2) / num_columns

    likelihood_object = computeNegativeLikelihood(nrow_Fk                   = nrow_Fk,
                                                  ncol_Fk                   = ncol_Fk,
                                                  s                         = s,
                                                  p                         = num_columns,
                                                  matrix_JSJ                = matrix_JSJ,
                                                  sample_covariance_trace   = sample_covariance_trace,
                                                  vfixed                    = None,
                                                  ldet                      = 0.0,
                                                  dtype                     = dtype,
                                                  device                    = device
                                                  )

    negative_log_likelihood = likelihood_object["negative_log_likelihood"]

    if onlylogLike:
        return {"negloglik": negative_log_likelihood}

    P = likelihood_object["P"]
    d_hat = likelihood_object["d_hat"]
    v = likelihood_object["v"]

    M = inverse_square_root_matrix @ P @ (P.T * d_hat[:, None]) @ inverse_square_root_matrix

    if not wSave:
        return {"v": v, 
                "M": M, 
                "s": s, 
                "negloglik": negative_log_likelihood
                }

    L = Fk @ ((torch.diag(torch.sqrt(d_hat)) @ P.T) @ inverse_square_root_matrix).T

    if ncol_Fk > 2:
        reduced_columns = torch.cat([
            torch.tensor([0], dtype=torch.int64, device=device),
            (d_hat[1:(ncol_Fk - 1)] > 0).nonzero(as_tuple=True)[0]
        ])
    else:
        reduced_columns = torch.tensor([ncol_Fk - 1], dtype=torch.int64, device=device)

    L = L[:, reduced_columns]

    invD = torch.ones(nrow_Fk, dtype=dtype, device=device) / (s + v)
    iDZ = invD[:, None] * data

    right = L @ (torch.linalg.inv(torch.eye(L.shape[1], dtype=dtype, device=device) + L.T @ (invD[:, None] * L)) @ (L.T @ iDZ))

    INVtZ = iDZ - invD[:, None] * right
    etatt = M @ Fk.T @ INVtZ

    GM = Fk @ M

    diag_matrix = (s + v) * torch.eye(nrow_Fk, dtype=dtype, device=device)

    V = M - GM.T @ invCz(R      = diag_matrix,
                         L      = L, 
                         z      = GM,
                         dtype  = dtype,
                         device = device
                         ).T

    return {"v": v,
            "M": M,
            "s": s,
            "negloglik": negative_log_likelihood,
            "w": etatt,
            "V": V
            }

# using in indeMLE
# check = ok, but have some problem
def EM0miss(
    Fk: torch.Tensor, 
    data: torch.Tensor, 
    Depsilon: torch.Tensor, 
    maxit: int=100, 
    avgtol: float=1e-4, 
    wSave: bool=False, 
    DfromLK: dict=None,
    vfixed: float=None,
    verbose: bool=True,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> dict:
    """

    """
    O = ~torch.isnan(data)
    TT = data.shape[1]
    ncol_Fk = Fk.shape[1]

    ziDz = torch.full((TT,), float('nan'), device=device)
    ziDB = torch.full((TT, ncol_Fk), float('nan'), device=device)
    db = {}
    D = Depsilon
    iD = torch.linalg.inv(D)
    diagD = isDiagonal(D)

    if DfromLK is not None:
        DfromLK = to_tensor(DfromLK, dtype=dtype, device=device)
        pick = DfromLK.get("pick", None)
        weights = DfromLK["weights"]
        if pick is None:
            pick = torch.arange(len(weights), dtype=torch.int64, device=device)
        else:
            pick = to_tensor(pick, dtype=torch.int64, device=device)
        weight = weights[pick]
        DfromLK["wX"] = DfromLK["wX"][pick, :]
        wwX = torch.diag(torch.sqrt(weight)) @ DfromLK["wX"]
        lQ = DfromLK["lambda"] * DfromLK["Q"]

    for tt in range(TT):
        if DfromLK is not None:
            obs_idx = O[:, tt].bool()
            iDt = None
            if obs_idx.sum() == O.shape[0]:
                wXiG = wwX @ torch.linalg.inv(DfromLK["G"])
            else:
                wX_obs = DfromLK["wX"][obs_idx, :]
                G = wX_obs.T @ wX_obs + lQ
                wXiG = wwX[obs_idx, :] @ torch.linalg.inv(G)

            Bt = Fk[obs_idx, :]
            if Bt.ndim == 1:
                Bt = Bt.unsqueeze(0)

            iDBt = weight[obs_idx].unsqueeze(1) * Bt - wXiG @ (wwX[obs_idx, :].T @ Bt)
            zt = data[obs_idx, tt]
            ziDz[tt] = torch.sum(zt * (weight[obs_idx] * zt - wXiG @ (wwX[obs_idx, :].T @ zt)))
            ziDB[tt, :] = (zt @ iDBt).squeeze()
            BiDBt = Bt.T @ iDBt

        else:
            if not diagD:
                iDt = torch.linalg.inv(D[obs_idx][:, obs_idx])
            else:
                iDt = iD[obs_idx][:, obs_idx]

            Bt = Fk[obs_idx, :]
            if Bt.ndim == 1:
                Bt = Bt.unsqueeze(0)

            iDBt = iDt @ Bt
            zt = data[obs_idx, tt]
            ziDz[tt] = torch.sum(zt * (iDt @ zt))
            ziDB[tt, :] = (zt @ iDBt).squeeze()
            BiDBt = Bt.T @ iDBt

        db[tt] = {"iDBt": iDBt,
                  "zt": zt,
                  "BiDBt": BiDBt
                  }

    del iDt, Bt, iDBt, zt, BiDBt
    _ = gc.collect()
    torch.cuda.empty_cache()

    dif = float("inf")
    cnt = 0
    Z0 = data.clone()
    Z0[torch.isnan(Z0)] = 0
    old = cMLEimat(Fk           = Fk, 
                   data         = Z0, 
                   s            = 0, 
                   wSave        = True,
                   S            =  None,
                   onlylogLike  =  None,
                   dtype        = dtype,
                   device       = device
                   )
    if vfixed is None:
        old["s"] = old["v"]
    else:
        old["s"] = to_tensor(vfixed)
    old["M"] = convertToPositiveDefinite(mat    = old["M"],
                                         dtype  = dtype,
                                         device = device
                                         )
    Ptt1 = old["M"]

    while (dif > (avgtol * (100 * (ncol_Fk ** 2)))) and (cnt < maxit):
        etatt = torch.zeros((ncol_Fk, TT), dtype=dtype, device=device)
        sumPtt = torch.zeros((ncol_Fk, ncol_Fk), dtype=dtype, device=device)
        s1 = torch.zeros(TT, dtype=dtype, device=device)

        for tt in range(TT):            
            ginv_Ptt1 = torch.linalg.pinv(convertToPositiveDefinite(mat     = Ptt1,
                                                                    dtype   = dtype,
                                                                    device  = device
                                                                    ))
            iP = convertToPositiveDefinite(mat      = ginv_Ptt1 + BiDBt / old["s"],
                                           dtype    = dtype,
                                           device   = device
                                           )
            Ptt = torch.linalg.inv(iP)  # will broken under some situation  # need fix
            Gt = (Ptt @ iDBt.T) / old["s"]
            eta = Gt @ zt
            s1kk = torch.diagonal(BiDBt @ (eta.unsqueeze(1) @ eta.unsqueeze(0) + Ptt))
            
            sumPtt += Ptt
            etatt[:, tt] = eta
            s1[tt] = torch.sum(s1kk)

        if vfixed is None:
            s = torch.max(
                (torch.sum(ziDz) - 2 * torch.sum(ziDB * etatt.T) + torch.sum(s1)) / torch.sum(O),
                torch.tensor(1e-8, dtype=dtype, device=device)
            )
            new = {"M": (etatt @ etatt.T + sumPtt) / TT,
                   "s": s,
                   }
        else:
            new = {"M": (etatt @ etatt.T + sumPtt) / TT,
                   "s": vfixed,
                   }

        new["M"] = (new["M"] + new["M"].T) / 2
        dif = torch.sum(torch.abs(new["M"] - old["M"])) + torch.abs(new["s"] - old["s"])
        cnt += 1
        old = new
        Ptt1 = old["M"]

    if verbose:
        info_msg = f'Number of iteration: {cnt}'
        LOGGER.info(info_msg)
        
    n2loglik = computeLikelihood(data       = data,
                                 Fk         = Fk,
                                 M          = new["M"],
                                 s          = new["s"],
                                 Depsilon   = Depsilon,
                                 dtype      = dtype,
                                 device     = device
                                 )

    if not wSave:
        return {
            "M": new["M"],
            "s": new["s"],
            "negloglik": n2loglik
        }

    elif DfromLK is not None:
        out = {
            "M": new["M"],
            "s": new["s"],
            "negloglik": n2loglik,
            "w": etatt,
            "V": new["M"] - (etatt @ etatt.T) / TT
        }

        eigenvalues, eigenvectors = torch.linalg.eigh(new["M"])
        L = Fk @ eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0.0)))

        weight = DfromLK["weights"][pick]
        wlk = torch.full((lQ.shape[0], TT), float("nan"), device=device)

        for tt in range(TT):
            obs_idx = O[:, tt].bool()
            if torch.sum(obs_idx) == O.shape[0]:
                wXiG = wwX @ torch.linalg.solve(DfromLK["G"], torch.eye(DfromLK["G"].shape[0], dtype=dtype, device=device))
            else:
                wX_tt = DfromLK["wX"][obs_idx]
                G = wX_tt.T @ wX_tt + lQ
                wXiG = wwX[obs_idx] @ torch.linalg.solve(G, torch.eye(G.shape[0], dtype=dtype, device=device))

            dat = data[obs_idx, tt]
            Lt = L[obs_idx]
            iDL = weight[obs_idx].unsqueeze(1) * Lt - wXiG @ (wwX[obs_idx].T @ Lt)
            itmp = torch.linalg.solve(
                torch.eye(L.shape[1], dtype=dtype, device=device) + (Lt.T @ iDL) / out["s"],
                torch.eye(L.shape[1], dtype=dtype, device=device)
            )
            iiLiD = itmp @ (iDL.T / out["s"])
            wlk[:, tt] = (wXiG.T @ dat - wXiG.T @ Lt @ (iiLiD @ dat)).squeeze()

        out["pinfo"] = {
            "wlk": wlk, 
            "pick": pick
        }
        out["missing"] = {
            "miss": 1 - O, 
            "maxit": maxit, 
            "avgtol": avgtol
        }
        return out

    else:
        out = {
            "M": new["M"],
            "s": new["s"],
            "negloglik": n2loglik,
            "w": etatt,
            "V": new["M"] - (etatt @ etatt.T) / TT
        }
        out["missing"] = {
            "miss": 1 - O, 
            "maxit": maxit, 
            "avgtol": avgtol
        }
        return out

# using in indeMLE
# check = ok
def cMLEsp(
    Fk: torch.Tensor,
    data: torch.Tensor,
    Depsilon: torch.Tensor,
    wSave: bool = False,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> dict:
    """
    Internal function: cMLEsp

    Parameters:
        Fk (torch.Tensor): 
            (n × K) matrix of basis function values at observation locations.

        data (torch.Tensor): 
            (n × T) data matrix (can contain NaN).

        Depsilon (torch.Tensor): 
            (n × n) diagonal covariance matrix (measurement error variances).

        wSave (bool, default=False): 
            Whether to compute and return weight and covariance matrices.

        device (torch.device or str, optional): 
            Device for computation.

        dtype (torch.dtype, optional): 
            Data precision.

    Returns:
        dict
            Dictionary containing:
            - 'M', 's', and (if wSave=True) 'w', 'V'
    """
    iD = torch.linalg.inv(Depsilon)
    ldetD = logDeterminant(mat = Depsilon).item()
    iDFk = iD @ Fk
    num_columns = data.shape[1]

    projection = computeProjectionMatrix(Fk1    = Fk,
                                         Fk2    = iDFk,
                                         data   = data,
                                         S      = None,
                                         dtype  = dtype,
                                         device = device
                                         )
    inverse_square_root_matrix = projection["inverse_square_root_matrix"]
    matrix_JSJ = projection["matrix_JSJ"]

    trS = torch.sum((iD @ data) * data) / num_columns
    out = cMLE(Fk                           = Fk,
               num_columns                  = num_columns,
               sample_covariance_trace      = trS,
               inverse_square_root_matrix   = inverse_square_root_matrix,
               matrix_JSJ                   = matrix_JSJ,
               s                            = 0,
               ldet                         = ldetD,
               wSave                        = wSave,
               onlylogLike                  = None,
               vfixed                       = None,
               dtype                        = dtype,
               device                       = device
               )

    if wSave:
        L = out["L"]
        s_plus_v = out["s"] + out["v"]
        invD = iD / s_plus_v
        iDZ = invD @ data
        right0 = L @ torch.linalg.solve(
            torch.eye(L.shape[1], dtype=dtype, device=device) + L.T @ (invD @ L),
            torch.eye(L.shape[1], dtype=dtype, device=device)
        )

        INVtZ = iDZ - invD @ right0 @ (L.T @ iDZ)
        etatt = out["M"] @ Fk.T @ INVtZ
        out["w"] = etatt
        GM = Fk @ out["M"]
        iDGM = invD @ GM
        out["V"] = out["M"] - GM.T @ (iDGM - invD @ right0 @ (L.T @ iDGM))

    out["s"] = out["v"]
    out.pop("v", None)
    out.pop("L", None)
    return out

# using in indeMLE
# check = ok
def cMLElk(
    Fk: torch.Tensor,
    data: torch.Tensor,
    Depsilon: torch.Tensor,
    wSave: bool = False,
    DfromLK: dict = None,
    vfixed: float = None,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> dict:
    """
    Internal function: cMLElk

    Parameters:
        Fk (torch.Tensor):
            (n × K) matrix of basis function values, where each column
            represents a basis function evaluated at the observation locations.

        data (torch.Tensor):
            (n × T) data matrix (can contain NaN) with z[t] as the t-th column.

        Depsilon (torch.Tensor):
            (n × n) diagonal covariance matrix of measurement errors.

        wSave (bool, default=False):
            Whether to compute and return additional weight and covariance matrices.

        DfromLK (dict):
            Dictionary containing precomputed quantities from the low-rank kernel step:
                - 'lambda' (float): regularization parameter.
                - 'pick' (list[int]): indices of selected observations.
                - 'wX' (torch.Tensor): weighted design matrix.
                - 'weights' (torch.Tensor): vector of weights.
                - 'Q' (torch.Tensor): penalty matrix.

        vfixed (float, optional):
            Fixed variance parameter (if provided, overrides estimation).

    Returns:
        dict:
            Dictionary containing:
                - 'M' (torch.Tensor): estimated model matrix.
                - 's' (float): variance parameter.
                - 'w' (torch.Tensor, optional): estimated weights if wSave=True.
                - 'V' (torch.Tensor, optional): covariance matrix of weights.
                - 'pinfo' (dict, optional): diagnostic info with 'wlk' and 'pick'.
    """
    num_columns = data.shape[1]
    lambda_ = DfromLK["lambda"]
    pick = DfromLK["pick"]
    wX = DfromLK["wX"]
    weight = DfromLK["weights"]
    Q = DfromLK["Q"]

    if len(pick) < wX.shape[0]:
        wX = wX[pick, :]
        weight = weight[pick]

    G = wX.T @ wX + lambda_ * Q
    wwX = torch.diag(torch.sqrt(weight)) @ wX
    wXiG = wwX @ torch.linalg.solve(G, torch.eye(G.shape[0], dtype=dtype, device=device))
    iDFk = weight.unsqueeze(1) * Fk - wXiG @ wwX.T @ Fk

    projection = computeProjectionMatrix(Fk1    = Fk,
                                         Fk2    = iDFk,
                                         data   = data,
                                         S      = None,
                                         dtype  = dtype,
                                         device = device
                                         )
    inverse_square_root_matrix = projection["inverse_square_root_matrix"]
    matrix_JSJ = projection["matrix_JSJ"]
    iDZ = weight.unsqueeze(1) * data - wXiG @ (wwX.T @ data)
    trS = torch.sum(iDZ * data) / num_columns
    ldetD = (
        -Q.shape[0] * torch.log(torch.tensor(lambda_, device=device))
        + logDeterminant(mat = G)
        - logDeterminant(mat = Q)
        - torch.sum(torch.log(weight))
    ).item()

    out = cMLE(Fk                           = Fk,
               num_columns                  = num_columns,
               sample_covariance_trace      = trS,
               inverse_square_root_matrix   = inverse_square_root_matrix,
               matrix_JSJ                   = matrix_JSJ,
               s                            = 0,
               ldet                         = ldetD,
               wSave                        = True,
               onlylogLike                  = False,
               vfixed                       = vfixed,
               dtype                        = dtype,
               device                       = device
               )
    L = out["L"]
    out["s"] = out["v"]
    out.pop("v", None)
    out.pop("L", None)
    if not wSave:
        return out

    iDL = weight.unsqueeze(1) * L - wXiG @ (wwX.T @ L)
    itmp = torch.linalg.solve(
        torch.eye(L.shape[1], dtype=dtype, device=device) + (L.T @ iDL) / out["s"],
        torch.eye(L.shape[1], dtype=dtype, device=device),
    )
    iiLiD = itmp @ (iDL.T / out["s"])
    MFiS11 = (out["M"] @ (iDFk.T / out["s"]) - ((out["M"] @ (iDFk.T / out["s"])) @ L) @ iiLiD)
    out["w"] = MFiS11 @ data
    out["V"] = MFiS11 @ (Fk @ out["M"])
    wlk = wXiG.T @ data - wXiG.T @ L @ (iiLiD @ data)

    out["pinfo"] = {"wlk": wlk,
                    "pick": pick
                    }

    return out




"""
Title: Setup file of autoFRK-Python Project
Author: Yao-Chih Hsu
Version: 1141018
Reviewer: Yao-Chih Hsu
Reviewed Version: 1141018
Description: Provides functions about `LatticeKrig` package which in R Language.
Reference: None
"""

# import modules
import torch
import numpy as np
from typing import Optional, Dict, Union
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize_scalar
from autoFRK.utils.utils import to_tensor
from autoFRK.utils.helper import logDeterminant
from autoFRK.utils.matrix_operator import getInverseSquareRootMatrix
from autoFRK.utils.estimator import cMLE

# using in autoFRK
# check = none
def initializeLKnFRK(
    data: torch.Tensor,
    location: torch.Tensor,
    nlevel: int = 3,
    weights: list = None,
    n_neighbor: int = 3,
    nu: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Initialize the hierarchical multi-resolution structure for FRK (Fixed Rank Kriging),
    handling missing data, imputing via nearest neighbors, and computing level weights and geometric type.

    Parameters:
    -----------
    data : torch.Tensor
        A tensor of shape (n, T) representing the observed data matrix.
        Each column corresponds to a time point, and rows correspond to spatial locations.
        Missing values are allowed (torch.nan).
    location : torch.Tensor
        A tensor of shape (n, d) specifying spatial coordinates for each observation.
        `d` is typically 1 (line), 2 (surface), or 3 (volume).
    nlevel : int, optional, default=3
        Number of resolution levels in the multi-resolution basis hierarchy.
        Each level corresponds to a distinct spatial scale used for FRK basis construction.
    weights : torch.Tensor, np.ndarray, or list, optional
        Optional weight vector or diagonal weight matrix of length/size `n`.
        If not provided, all weights are set to 1.
    n_neighbor : int, optional, default=3
        Number of nearest neighbors used for imputing missing data.
        Missing entries are replaced with the average of their `n_neighbor` nearest observed values.
    nu : int, optional, default=1
        Smoothness parameter controlling the relative contribution of fine and coarse scales.
        Used in the computation of the level weights `alpha`.

    Returns:
    --------
    dict
        - **x** (torch.Tensor): filtered location matrix after removing empty rows.
        - **z** (torch.Tensor): data matrix after imputation and filtering.
        - **n** (torch.Tensor): number of valid spatial locations.
        - **alpha** (torch.Tensor): normalized weights for each resolution level.
        - **gtype** (str): geometry type ("LKInterval", "LKRectangle", or "LKBox").
        - **weights** (torch.Tensor): weight vector for spatial locations.
        - **nlevel** (int): number of hierarchical levels.
        - **location** (torch.Tensor): original location matrix (possibly with NA rows removed).
        - **pick** (torch.Tensor): indices (1-based, to match R) of retained rows from the original data.
    """
    dtype=data.dtype
    device=data.device

    data = data.detach().cpu().numpy()
    location = location.detach().cpu().numpy()

    non_empty = ~np.all(np.isnan(data), axis=0)
    data = data[:, non_empty]

    valid_rows = ~np.all(np.isnan(data), axis=1)
    data = data[valid_rows, :]
    x = location[valid_rows, :]
    pick = np.where(valid_rows)[0]

    # Impute missing values using nearest neighbors
    nas = np.isnan(data).sum()
    if nas > 0:
        for tt in range(data.shape[1]):
            where = np.isnan(data[:, tt])
            if not np.any(where):
                continue
            cidx = np.where(~where)[0]
            nbrs = NearestNeighbors(n_neighbors=n_neighbor).fit(x[cidx, :])
            distances, nn_index = nbrs.kneighbors(x[where, :])
            nn_index = cidx[nn_index]
            nn_values = data[nn_index, tt]
            data[where, tt] = np.nanmean(nn_values, axis=1)

    z = np.asarray(data)
    n, d = x.shape

    if d == 1:
        gtype = "LKInterval"
    elif d == 2:
        gtype = "LKRectangle"
    else:
        gtype = "LKBox"

    thetaL = 2.0 ** (-1 * np.arange(1, nlevel + 1))
    alpha = thetaL ** (2 * nu)
    alpha = alpha / np.sum(alpha)

    if weights is None:
        weights = np.ones(n)

    out = {"x": x,
           "z": z,
           "n": n,
           "alpha": alpha,
           "gtype": gtype,
           "weights": weights,
           "nlevel": nlevel,
           "location": location,
           "pick": pick
           }
    out = to_tensor(out, dtype=dtype, device=device)
    return out

# using in autoFRK
# check = none
def setLKnFRKOption(
    LK_obj: dict,
    Fk: torch.Tensor,
    nc: Optional[torch.Tensor] = None,
    Ks: Optional[int] = None,
    a_wght: Optional[float] = None,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = 'cpu'
) -> dict:
    """
    Internal function: setLKnFRKOption

    Parameters
    ----------
    LK_obj : dict
        A dictionary produced from `initializeLKnFRK()`, containing 'x', 'z', 'alpha', etc.
    Fk : torch.Tensor
        n x K basis function matrix. Each column is a basis function evaluated at locations.
    nc : torch.Tensor, optional
        Numeric matrix/vector produced by `setNC()`. Default is None.
    Ks : int, optional
        Number of basis functions. Default is Fk.shape[1].
    a_wght : float, optional
        Scalar weight used in basis construction. Default is None (will be set to 2*d + 0.01).

    Returns
    -------
    dict
        A dictionary containing:
        - DfromLK : dict with keys Q, weights, wX, G, lambda, pick
        - s : estimated scale parameter
        - LKobj : dictionary containing summary, par.grid, LKinfo.MLE, lnLike.eval, lambda.MLE, call, taskID
    """
    x = LK_obj['x']
    z = LK_obj['z']
    alpha = LK_obj['alpha']
    alpha = alpha / alpha.sum()
    gtype = LK_obj['gtype']
    weights = LK_obj['weights']
    if len(LK_obj['pick']) < len(weights):
        weights = weights[LK_obj['pick']]
    nlevel = LK_obj['nlevel']
    TT = z.shape[1]
    Fk = Fk[LK_obj['pick'], :]

    if nc is None:
        nc = setNC(z,
                   x,
                   nlevel
                   )
    if a_wght is None:
        a_wght = 2 * x.shape[1] + 0.01

    info = LKrigSetup(x         = x,
                      a_wght    = a_wght,
                      nlevel    = nlevel,
                      NC        = nc,
                      alpha     = alpha,
                      LKGeometry= gtype,
                      lambda_   = 1.0
                      )              

    location = x
    phi = LKrig_basis(location,
                      info
                      )
    w = torch.diag(torch.sqrt(weights))
    wX = w @ phi
    wwX = w @ wX
    XwX = wX.T @ wX

    Qini = LKrig_precision(info)

    def iniLike(par,
                data=z,
                full=False
    ) -> Union[dict, torch.Tensor]:
        """
        inner function ...
        """
        lambda_ = torch.exp(torch.tensor(par, dtype=dtype, device=device))
        G = XwX + lambda_ * Qini
        wXiG = wwX @ torch.linalg.inv(G)
        iDFk = weights * Fk - wXiG @ (wwX.T @ Fk)
        iDZ = weights * data - wXiG @ (wwX.T @ data)
        ldetD = -Qini.shape[0] * torch.log(lambda_) + logDeterminant(mat = G_mat)
        trS = torch.sum(iDZ * data) / TT
        half = getInverseSquareRootMatrix(Fk, iDFk)
        ihFiD = half @ iDFk.T
        LSL = (ihFiD @ data) @ (ihFiD @ data).T / TT
        if not full:
            return cMLE(Fk,
                        TT,
                        trS,
                        half,
                        LSL,
                        s=0,
                        ldet=ldetD,
                        wSave=False
                        )['negloglik']
        else:
            llike = ldetD - logDeterminant(mat = Qini) - torch.log(weights).sum()
            return cMLE(Fk,
                        TT,
                        trS,
                        half,
                        LSL,
                        s=0,
                        ldet=llike,
                        wSave=True,
                        onlylogLike=False,
                        vfixed=None
                        )

    sol = minimize_scalar(iniLike,
                          bounds=(-16, 16),
                          method="bounded",
                          options={"xatol": np.finfo(float).eps ** 0.025}
                          )
    lambda_MLE = to_tensor(sol.x, dtype=dtype, device=device)
    out = iniLike(to_tensor(sol.x, dtype=dtype, device=device),
                  z,
                  full=True
                  )
    llike = out['negloglik']
    info_MLE = LKrigSetup(x=x,
                          a_wght=a_wght,
                          nlevel=nlevel,
                          NC=nc,
                          alpha=alpha,
                          LKGeometry=gtype,
                          lambda_=lambda_MLE
                          )
    info_MLE['llike'] = llike
    info_MLE['time'] = None
    Q = LKrig_precision(info_MLE)
    G_mat = wX.T @ wX + info_MLE['lambda_'] * Q

    ret =   {
                "DfromLK": {
                    "Q": Q,
                    "weights": weights,
                    "wX": wX,
                    "G": G_mat,
                    "lambda": info_MLE['lambda_'],
                    "pick": LK_obj['pick']
                },
                "s": out['v'],
                "LKobj": {
                    "summary": None,
                    "par_grid": None,
                    "LKinfo_MLE": info_MLE,
                    "lnLike_eval": None,
                    "lambda_MLE": info_MLE['lambda_'],
                    "call": None,
                    "taskID": None
                }
            }
    return to_tensor(ret, dtype=dtype, device=device)

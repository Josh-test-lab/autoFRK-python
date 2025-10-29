"""
Title: Predictor of autoFRK-Python Project
Author: Yao-Chih Hsu
Version: 1141025
Description: This file contain prediction-related functions for the autoFRK-Python project.
Reference: `autoFRK` R package by Wen-Ting Wang from https://github.com/egpivo/autoFRK
"""

# import modules
import torch
from typing import Optional, Union, Dict, Tuple
from ..utils.utils import to_tensor
from ..utils.logger import LOGGER
from ..utils.device import check_device
from ..utils.matrix_operator import invCz, decomposeSymmetricMatrix, to_sparse
from ..mrts import create_rectangular_tps_matrix, predict_rectangular_tps_matrix

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
    tps_method: str | int | None = None,
    dtype: torch.dtype=torch.float64,
    device: Optional[Union[torch.device, str]] = 'cpu'
) -> Dict[str, Union[torch.Tensor, None]]:
    """
    Predict values and estimate of standard errors based on Fixed Rank Kriging (autoFRK) model.

    This function provides predictions at specified locations and times based on an `autoFRK` 
    model object, optionally including standard errors.

    Parameters
    ----------
    obj : dict
        Model object obtained from `autoFRK`.
    obsData : torch.Tensor, optional
        Observed data used for prediction. Default is None, which uses `Data` from `obj`.
    obsloc : torch.Tensor, optional
        Coordinates of observation locations for `obsData`. Only applicable for models
        fitted using `MRTS` basis functions. Default is None, which uses `loc` from `obj`.
    mu_obs : float or torch.Tensor, optional
        Deterministic mean values at observation locations. Default is 0.
    newloc : torch.Tensor, optional
        Coordinates of new locations for prediction. Default is None, which predicts at observed locations.
    basis : torch.Tensor, optional
        Basis matrix at `newloc`. Can be omitted if model uses default `MRTS` basis functions.
    mu_new : float or torch.Tensor, optional
        Deterministic mean values at `newloc`. Default is 0.
    se_report : bool, optional
        If True, returns standard errors of predictions. Default is False.
    tps_method : str, int or None, optional
        Specifies the method used to compute thin-plate splines (TPS). Default is None.
        Options:
            - None: Auto detect by `forward` method.
            - "rectangular" (or 0): Compute TPS in Euclidean (rectangular) coordinates.
            - "spherical_fast" (or 1): Use spherical coordinates but apply the rectangular TPS formulation for faster computation.
            - "spherical" (or 2): Compute TPS directly in spherical coordinates.
    dtype : torch.dtype, optional
        Desired torch dtype for computation (default: torch.float64).
    device : torch.device or str, optional
        Target device for computation (default: 'cpu').

    Returns
    -------
    dict
        Dictionary with the following keys:
        - **pred.value** (`torch.Tensor`): Predicted values, shape (num_locations, num_times).
        - **se** (`torch.Tensor` or None): Standard errors of predictions, only if `se_report=True`.
    """
    obj = to_tensor(obj     = obj,
                    dtype   = dtype,
                    device  = device
                    )
    
    # check tps_method
    if tps_method is None:
        if obj.get('tps_method', None) is not None:
            tps_method = obj['tps_method']
        else:
            error_msg = f'Could not find the parameter "tps_method". Please specify a valid method ("rectangular", "spherical_fast" or "spherical").'
            LOGGER.error(error_msg)
            ValueError(error_msg)
    if not isinstance(tps_method, str):
        tps_method = int(tps_method)
    if tps_method == 0 or tps_method == "rectangular":
        tps_method = "rectangular"
    elif tps_method == 1 or tps_method == "spherical_fast":
        tps_method = "spherical_fast"
    elif tps_method == 2 or tps_method == "spherical":
        tps_method = "spherical"
    else:
        error_msg = f'Invalid tps_method "{tps_method}", it should be one of "rectangular", "spherical_fast", or "spherical".'
        LOGGER.error(error_msg)
        ValueError(error_msg)

    # check device
    device = check_device(obj   = obj,
                          device= device
                          )

    if basis is None:
        if newloc is None:
            if "G" not in obj:
                error_msg = f"Basis matrix of new locations should be given (unless the model was fitted with mrts bases)!"
                LOGGER.error(error_msg)
                raise ValueError(error_msg)
            basis = obj["G"]["MRTS"]

        else:
            newloc = to_tensor(obj      = newloc,
                               dtype    = dtype,
                               device   = device
                               )
            basis = predict_MRTS(obj        = obj["G"],
                                 newx       = newloc,
                                 tps_method = tps_method,
                                 dtype      = dtype,
                                 device     = device
                                 )

    if basis.ndim == 1:
        basis = to_tensor(obj   = basis,
                          dtype = dtype,
                          device= device
                          )
        basis = basis.unsqueeze(0)

    if obsloc is None:
        nobs = obj["G"]["MRTS"].shape[0]
    else:
        if obsloc.ndim == 1:
            obsloc = obsData.reshape(-1, 1)
        nobs = to_tensor(obj    = obsloc.shape[0],  
                         dtype  = dtype,
                         device = device
                         )

    if obsData is not None:
        if obsData.ndim == 1:
            obsData = obsData.reshape(-1, 1)
        obsData -= mu_obs
        if obsData.shape[0] != nobs:
            error_msg = f"Dimensions of obsloc and obsData are not compatible!"
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

    if newloc is not None:
        if newloc.ndim == 1:
            newloc = newloc.reshape(-1, 1)
        if basis.shape[0] != newloc.shape[0]:
            error_msg = f"Dimensions of obsloc and obsData are not compatible!"
            LOGGER.error(error_msg)
            raise ValueError(error_msg)
    else:
        if basis.shape[0] != obj["G"]["MRTS"].shape[0]:
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
                    Fk = obj["G"]["MRTS"][pick]
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
                        V = M - GM.T @ invCz(R      = obj["s"] * De,
                                             L      = L,
                                             z      = GM,
                                             dtype  = dtype,
                                             device = device
                                             ).T
                        se[:, tt] = torch.sqrt(torch.clamp(torch.sum((basis @ V) * basis, dim=1), min=0.0))

        if obsData is not None:
            pick = (~torch.isnan(obsData)).nonzero(as_tuple=True)[0].tolist()
            if obsloc is None:
                De = pinfo["D"][pick][:, pick]
                G = obj["G"]["MRTS"][pick]
            else:
                De = torch.eye(len(pick), dtype=dtype, device=device)
                G = predict_MRTS(obj        = obj["G"],
                                 newx       = obsloc[pick],
                                 tps_method = tps_method,
                                 dtype      = dtype,
                                 device     = device
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
                                        ).T

            if se_report:
                V = M - GM.T @ invCz(R      = obj["s"] * De,
                                     L      = L,
                                     z      = GM,
                                     dtype  = dtype,
                                     device = device
                                     ).T
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
                G = predict_MRTS(obj["G"], newx=obsloc[pick, :])

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
def predict_MRTS(
    obj: dict,
    newx: Optional[torch.Tensor] = None,
    tps_method: str = "rectangular",
    dtype: torch.dtype=torch.float64,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Evaluate multi-resolution thin-plate spline basis functions at given locations.

    This function provides a generic prediction method for `autoFRK` model objects.

    Parameters
    ----------
    obj : dict
        Object produced from calling `G`. Must contain:
        - 'Xu': (n × d) tensor of knot locations
        - 'nconst': normalization constants (1D tensor)
        - 'BBBH': precomputed thin-plate spline matrix
        - 'UZ': orthogonal basis matrix
        - 'MRTS': evaluated multi-resolution thin-plate spline basis basis matrix
    newx : torch.Tensor, optional
        (n × d) tensor of coordinates of new locations.
        If None, returns `obj['MRTS']`.
    tps_method : str, optional
        Specifies the method used to compute thin-plate splines (TPS). Default is "rectangular".
        Options:
            - "rectangular": Compute TPS in Euclidean (rectangular) coordinates.
            - "spherical_fast": Use spherical coordinates but apply the rectangular TPS formulation for faster computation.
    dtype : torch.dtype, optional
        Desired torch dtype for computation (default: torch.float64).
    device : torch.device or str, optional
        Target device for computation (default: 'cpu').

    Returns
    -------
    torch.Tensor
        (n × k) tensor of k MRTS basis function values evaluated at `newx`.
    """
    if newx is None:
        return obj["MRTS"]

    Xu = obj["Xu"]
    k = obj["MRTS"].shape[1]
    if tps_method in ("rectangular", "spherical_fast"):
        n = Xu.shape[0]
        xobs_diag = torch.diag(torch.sqrt(torch.tensor(n / (n - 1), dtype=dtype, device=device)) / Xu.std(dim=0, unbiased=True))
        ndims = Xu.shape[1]
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
                                      tps_method= tps_method,
                                      dtype     = dtype,
                                      device    = device
                                      )["X1"]
            
            X1 = X1[:, :kstar]
            return torch.cat([X2, X1], dim=1)
        else:
            return X2
        
    elif tps_method == "spherical":
        from ..mrts import compute_mrts_spherical
        res = compute_mrts_spherical(knot     = Xu,
                                     k        = k,
                                     X        = newx,
                                     dtype    = dtype,
                                     device   = device,
                                     )
        return res 

    else:
        error_msg = f'Invalid tps_method "{tps_method}", it should be one of "rectangular", "spherical_fast", or "spherical".'
        LOGGER.error(error_msg)
        ValueError(error_msg)


# using in predict_MRTS
# check = none
def predictMrtsWithBasis(
    s: torch.Tensor,
    xobs_diag: torch.Tensor,
    s_new: torch.Tensor,
    BBBH: torch.Tensor,
    UZ: torch.Tensor,
    nconst: torch.Tensor,
    k: int,
    tps_method: str = "rectangular",
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> Dict[str, torch.Tensor]:
    """
    Predict MRTS basis values at new locations.

    This function computes the thin-plate spline basis for new locations (`s_new`) 
    using precomputed matrices and transformations from a fitted MRTS model.

    Parameters
    ----------
    s : torch.Tensor
        Original location matrix, shape (n, d).
    xobs_diag : torch.Tensor
        Observation-related matrix, shape (n, n).
    s_new : torch.Tensor
        New location matrix for prediction, shape (n2, d).
    BBBH : torch.Tensor
        Precomputed internal matrix for transformation, shape (d + 1, k).
    UZ : torch.Tensor
        Orthogonal basis matrix from the fitted MRTS, shape (n, k).
    nconst : torch.Tensor
        Column mean vector, shape (d + 1, ).
    k : int
        Rank (number of basis functions used).
    tps_method : str, optional
        Specifies the method used to compute thin-plate splines (TPS). Default is "rectangular".
        Options:
            - "rectangular": Compute TPS in Euclidean (rectangular) coordinates.
            - "spherical_fast": Use spherical coordinates but apply the rectangular TPS formulation for faster computation.
    dtype : torch.dtype, optional
        Desired torch dtype for computation (default: torch.float64).
    device : torch.device or str, optional
        Target device for computation (default: 'cpu').

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing:
        - "X" (torch.Tensor): Original locations matrix `s`.
        - "UZ" (torch.Tensor): Orthogonal basis matrix `UZ`.
        - "BBBH" (torch.Tensor): Precomputed matrix `BBBH`.
        - "nconst" (torch.Tensor): Column means or normalization constants.
        - "X1" (torch.Tensor): Predicted basis matrix at `s_new`, adjusted by `BBBH` and `UZ`.
    """
    n, d = s.shape
    n2 = s_new.shape[0]
    Phi_new = predict_rectangular_tps_matrix(s_new      = s_new,
                                             s          = s,
                                             tps_method = tps_method,
                                             dtype      = dtype,
                                             device     = device
                                             )

    X1 = Phi_new @ UZ[:n, :k]
    B = torch.ones((n2, d + 1), dtype=dtype, device=device)
    B[:, -d:] = s_new

    return {"X": s,
            "UZ": UZ,
            "BBBH": BBBH,
            "nconst": nconst,
            "X1": X1 - B @ (BBBH @ UZ[:n, :k])
            }

# using in MRTS.forward
# predictMrts
# check = none
def predict_mrts_rectangular(
    s: torch.Tensor,
    xobs_diag: torch.Tensor,
    s_new: torch.Tensor,
    k: int,
    tps_method: str = "rectangular",
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Predict on new locations using the MRTS method.

    Computes the MRTS basis for new locations `s_new` given a fitted MRTS model 
    represented by matrices computed from original locations `s`.

    Parameters
    ----------
    s : torch.Tensor
        Original location matrix, shape (n, d).
    xobs_diag : torch.Tensor
        Observation-related matrix, typically diagonal or similar, shape (n, n).
    s_new : torch.Tensor
        New location matrix for prediction, shape (n2, d).
    k : int
        Rank (number of basis functions).
    tps_method : str, optional
        Specifies the method used to compute thin-plate splines (TPS). Default is "rectangular".
        Options:
            - "rectangular": Compute TPS in Euclidean (rectangular) coordinates.
            - "spherical_fast": Use spherical coordinates but apply the rectangular TPS formulation for faster computation.
    dtype : torch.dtype, optional
        Desired torch dtype for computation (default: torch.float64).
    device : torch.device or str, optional
        Target device for computation (default: 'cpu').

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing:
        - "X" (torch.Tensor): Core location matrix.
        - "UZ" (torch.Tensor): Orthogonal basis matrix.
        - "BBBH" (torch.Tensor): Transformed internal matrix for computation.
        - "nconst" (torch.Tensor): Column means or normalization constants.
        - "X1" (torch.Tensor): Predicted basis matrix at `s_new`, adjusted by transformations.
    """
    n, d = s.shape
    n2 = s_new.shape[0]

    # Update B, BBB, lambda, gamma
    Phi, B, BBB, lambda_, gamma = updateMrtsBasisComponents(s           = s,
                                                            k           = k,
                                                            tps_method  = tps_method,
                                                            dtype       = dtype,
                                                            device      = device
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
                                   dtype    = dtype,
                                   device   = device
                                   )

    # Create thin plate splines, Phi_new by new positions `s_new`
    Phi_new = predict_rectangular_tps_matrix(s_new      = s_new,
                                             s          = s,
                                             tps_method = tps_method,
                                             dtype      = dtype,
                                             device     = device
                                             )
    
    X1 = Phi_new @ UZ[:n, :k]
    B_new = torch.ones((n2, d + 1), dtype=dtype, device=device)
    B_new[:, -d:] = s_new

    return {"X":        X,
            "UZ":       UZ,
            "BBBH":     BBB @ Phi,
            "nconst":   nconst,
            "X1":       X1 - B_new @ ((BBB @ Phi) @ UZ[:n, :k])
            }

# using in predictMrts
# check = none
def updateMrtsBasisComponents(
    s: torch.Tensor,
    k: int,
    tps_method: str = "rectangular",
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute MRTS basis components via eigen-decomposition of the projected thin-plate spline matrix.

    Parameters
    ----------
    s : torch.Tensor
        Location matrix of shape (n, d), where n is the number of locations and d the dimension.
    k : int
        Number of leading eigenvalues/eigenvectors to compute (1 <= k <= n-1).
    tps_method : str, optional
        Specifies the method used to compute thin-plate splines (TPS). Default is "rectangular".
        Options:
            - "rectangular": Compute TPS in Euclidean (rectangular) coordinates.
            - "spherical_fast": Use spherical coordinates but apply the rectangular TPS formulation for faster computation.
    dtype : torch.dtype, optional
        Desired torch dtype for computation (default: torch.float64).
    device : torch.device or str, optional
        Target device for computation (default: 'cpu').

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        - Phi (torch.Tensor): Thin-plate spline matrix, shape (n, n)
        - B (torch.Tensor): Design matrix [1 | s], shape (n, d+1)
        - BBB (torch.Tensor): Projection matrix B(B'B)^{-1}, shape (d+1, n)
        - lambda (torch.Tensor): Leading k eigenvalues of the projected Phi matrix, shape (k,)
        - gamma (torch.Tensor): Corresponding eigenvectors, shape (n, k)
    """
    n, d = s.shape

    # Create thin plate splines Phi
    Phi = create_rectangular_tps_matrix(s           = s,
                                        tps_method  = tps_method,
                                        dtype       = dtype,
                                        device      = device
                                        )
    
    B = torch.ones((n, d + 1), dtype=dtype, device=device)
    B[:, -d:] = s
    Bt = B.t()
    BtB = Bt @ B
    BtB = (BtB + BtB.T) / 2

    L = torch.linalg.cholesky(BtB)
    # BBB := B(B'B)^{-1}B'
    BBB = torch.cholesky_solve(Bt, L)  # need fix
    # Phi_proj := \Phi((I-B(B'B)^{-1}B')
    Phi_proj = Phi - (Phi @ B) @ BBB
    # quadratic := ((I-B(B'B)^{-1}B')\Phi((I-B(B'B)^{-1}B')
    quadratic = Phi_proj - BBB.t() @ (Bt @ Phi_proj)

    # Set a convergence threshold for eigen-decomposition
    ncv = min(n, max(2 * k + 1, 20))
    lambda_, gamma = decomposeSymmetricMatrix(M     = quadratic,
                                              k     = k,
                                              ncv   = ncv,
                                              dtype = dtype,
                                              device= device
                                              )

    return Phi, B, BBB, lambda_, gamma

# using in predictMrts
# check = none
def updateMrtsCoreComponentX(
    s: torch.Tensor,
    gamma: torch.Tensor,
    k: int,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute MRTS core component X and normalization constants (nconst).

    Parameters
    ----------
    s : torch.Tensor, shape (n, d)
        Input location matrix with n samples in d dimensions.
    gamma : torch.Tensor, shape (n, k)
        Leading k eigenvectors of the projected thin-plate spline matrix.
    k : int
        Number of eigenvectors to use.
    dtype : torch.dtype, optional
        Desired torch dtype for computation (default: torch.float64).
    device : torch.device or str, optional
        Target device for computation (default: 'cpu').

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - X : torch.Tensor, shape (n, d + 1 + k)
            MRTS core component matrix combining bias, centered/scaled coordinates, and eigenvectors.
        - nconst : torch.Tensor, shape (d,)
            Normalization constants (column norms) used to scale the coordinates.
    """
    n, d = s.shape
    root = torch.sqrt(torch.tensor(float(n), dtype=dtype, device=device))
    X = torch.ones((n, k + d + 1), dtype=dtype, device=device)

    X_center = s - s.mean(dim=0, keepdim=True)
    nconst = torch.norm(X_center, dim=0)

    X[:n, 1:(d + 1)] = X_center * (root / nconst)
    X[:n, (d + 1):(d + 1 + k)] = gamma * root

    nconst = nconst / root

    return X, nconst

# using in predictMrts
# check = none
def updateMrtsCoreComponentUZ(
    s: torch.Tensor,
    xobs_diag: torch.Tensor,
    B: torch.Tensor,
    BBB: torch.Tensor,
    lambda_: torch.Tensor,
    gamma: torch.Tensor,
    k: int,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> torch.Tensor:
    """
    Compute MRTS core component UZ matrix.

    Parameters
    ----------
    s : torch.Tensor, shape (n, d)
        Input location matrix with n samples in d dimensions.
    xobs_diag : torch.Tensor, shape (d, d)
        Diagonal matrix used to scale observations.
    B : torch.Tensor, shape (n, d + 1)
        Design matrix of ones + coordinates.
    BBB : torch.Tensor, shape (d + 1, n)
        Matrix B(B^T B)^{-1} used for projections.
    lambda_ : torch.Tensor, shape (k,)
        Leading eigenvalues of the projected thin-plate spline matrix.
    gamma : torch.Tensor, shape (n, k)
        Leading eigenvectors of the projected thin-plate spline matrix.
    k : int
        Number of eigenvectors.
    dtype : torch.dtype, optional
        Desired torch dtype for computation (default: torch.float64).
    device : torch.device or str, optional
        Target device for computation (default: 'cpu').

    Returns
    -------
    torch.Tensor, shape (n + d + 1, k + d + 1)
        UZ matrix combining scaled/projected eigenvectors and observation diagonal.
    """
    n, d = s.shape
    root = torch.sqrt(torch.tensor(float(n), dtype=dtype, device=device))

    gammas = gamma - B @ (BBB @ gamma)
    gammas = gammas / lambda_.unsqueeze(0) * root

    UZ = torch.zeros((n + d + 1, k + d + 1), dtype=dtype, device=device)
    UZ[:n, :k] = gammas
    UZ[n, k] = 1.0
    UZ[(n + 1):(n + 1 + d), (k + 1):(k + 1 + d)] = xobs_diag

    return UZ

"""
Title: Automatic Fixed Rank Kriging
Author: Yao-Chih Hsu
Version: 1141025
Description: `autoFRK` is an R package to mitigate the intensive computation for modeling regularly/irregularly located spatial data using a class of basis functions with multi-resolution features and ordered in terms of their resolutions, and this project is to implement the `autoFRK` in Python.
Reference: Resolution Adaptive Fixed Rank Kringing by ShengLi Tzeng & Hsin-Cheng Huang
"""

# import modules
import torch
import torch.nn as nn
from typing import Optional, Union
from .utils.logger import LOGGER, set_logger_level
from .utils.device import setup_device, garbage_cleaner
from .utils.utils import to_tensor
from .utils.helper import fast_mode_knn_torch, fast_mode_knn_sklearn, fast_mode_knn_faiss, selectBasis
from .utils.estimator import indeMLE
from .utils.predictor import predict_FRK

# class AutoFRK
class AutoFRK(nn.Module):
    """
    Automatic Fixed Rank Kriging (autoFRK)

    This function performs resolution-adaptive Fixed Rank Kriging (FRK) based on 
    spatial data observed at one or multiple time points, using a hierarchical 
    multi-resolution basis and model-based estimation. The spatial process is modeled as:

        z[t] = μ + G @ w[t] + η[t] + e[t],
        w[t] ~ N(0, M),
        e[t] ~ N(0, s * D),
        for t = 1, ..., T

    where:
    - z[t]: observed data at n spatial locations,
    - μ: deterministic mean term,
    - G: spatial basis matrix,
    - w[t]: latent random effects,
    - η[t]: fine-scale process (optional),
    - D: covariance of measurement error.

    Methods
    -------
    __init__(dtype, device)
        Initialize model configuration, including computation device and precision.
    forward(data, loc, ...)
        Fit the FRK model to spatial data, estimating basis coefficients and covariance terms.
    predict(loc_new, ...)
        Predict at new spatial locations using the fitted FRK model.

    References
    ----------
    - Tzeng, S. & Huang, H.-C. (2018). *Resolution Adaptive Fixed Rank Kriging*. 
      Technometrics. https://doi.org/10.1080/00401706.2017.1345701  
    - Nychka, D., Hammerling, D., Sain, S., & Lenssen, N. (2016). 
      *LatticeKrig: Multiresolution Kriging Based on Markov Random Fields*.
    - Tzeng S, Huang H, Wang W, Nychka D, Gillespie C (2021). autoFRK: Automatic Fixed
      Rank Kriging_. R package version 1.4.3. https://CRAN.R-project.org/package=autoFRK
    """
    def __init__(
        self,
        logger_level: int | str=20,
        dtype: torch.dtype | None=None,
        device: Optional[Union[torch.device, str]]=None
        ):
        """
        Initialize an autoFRK model instance.

        Parameters
        ----------
        logger_level : int, str, optional
            Logging level for the process (e.g., logging.INFO or 20). Default is 20.
            Possible values:
            - `logging.NOTSET` or 0        : No specific level; inherits parent logger level
            - `logging.DEBUG`  or 10       : Detailed debugging information
            - `logging.INFO`   or 20       : General information about program execution
            - `logging.WARNING` or 30     : Warning messages, indicate potential issues
            - `logging.ERROR`  or 40       : Error messages, something went wrong
            - `logging.CRITICAL` or 50    : Severe errors, program may not continue
        dtype : torch.dtype or None, optional
            Data type for all internal tensors (default: None, auto detected).
        device : torch.device or str, optional
            Computation device ("cpu" or "cuda"). Automatically detected if None.
        """
        super().__init__()

        # set logger level
        if logger_level != 20:
            set_logger_level(LOGGER, logger_level)

        # setup device
        self.device = device

        # dtype check
        if dtype is not None and not isinstance(dtype, torch.dtype):
            error_msg = f"Invalid dtype: expected a torch.dtype instance, got {type(dtype).__name__}"
            LOGGER.error(error_msg)
            raise TypeError(error_msg)
        self.dtype = dtype

    def forward(
        self, 
        data: torch.Tensor, 
        loc: torch.Tensor,
        mu: Union[float, torch.Tensor]=0.0, 
        D: torch.Tensor=None, 
        G: torch.Tensor=None,
        maxK: int=None, 
        Kseq: torch.Tensor=None, 
        maxknot: int=5000,
        method: str="fast", 
        n_neighbor: int=3, 
        maxit: int=50, 
        tolerance: float=1e-6,
        requires_grad: bool=False,
        tps_method: str | int="rectangular",
        finescale: bool=False, 
        dtype: torch.dtype | None=None,
        device: Optional[Union[torch.device, str]]=None
    ) -> dict:
        """
        `autoFRK` forward method

        Perform model fitting and estimation for the autoFRK process.

        Parameters
        ----------
        data : torch.Tensor
            (n, T) data matrix of observed values. Each column corresponds to one time step.
            Missing values are allowed (`torch.nan`).
        loc : torch.Tensor
            (n, d) coordinate matrix specifying spatial locations.
        mu : float or torch.Tensor, optional
            Mean term (scalar or (n,) tensor). Default is 0.0.
        D : torch.Tensor, optional
            (n, n) covariance matrix of measurement errors. If None, identity is used.
        G : torch.Tensor, optional
            (n, K) matrix of basis functions evaluated at `loc`. If None, basis functions
        maxK : int, optional
            Maximum number of basis functions to consider. Default is `10 * sqrt(n)` if n > 100, else n.
        Kseq : torch.Tensor, optional
            Sequence of candidate numbers of basis functions to test. Default is None.
        maxknot : int, optional
            Maximum number of knots for multi-resolution TPS basis generation. Default is 5000.
        method : str, optional
            Method for estimation. Supported values:
            - `"fast"`: approximate imputation using nearest neighbors by PyTorch module (default)
            - `"EM"`: expectation-maximization
        n_neighbor : int, optional
            Number of neighbors used for "fast" imputation. Default is 3.
        maxit : int, optional
            Maximum number of iterations for optimization. Default is 50.
        tolerance : float, optional
            Convergence tolerance for iterative optimization. Default is 1e-6.
        requires_grad : bool, optional
            If True, enables gradient computation for `data` tensor. Default is False.
        tps_method : str or int, optional
            Specifies the method used to compute thin-plate splines (TPS). Default is "rectangular".
            Options:
                - "rectangular" (or 0): Compute TPS in Euclidean (rectangular) coordinates.
                - "spherical" (or 1): Compute TPS directly in spherical coordinates.
                - "spherical_fast" (or 2): Use spherical coordinates but apply the rectangular TPS formulation for faster computation.
        finescale : bool, optional
            Whether to include an approximate stationary fine-scale process η[t].
            When True, only the diagonal elements of D are used. Default is False.
        dtype : torch.dtype or None, optional
            Data type used in computations (e.g., `torch.float64`). Default is None (auto detected).
        device : torch.device or str, optional
            Target computation device ("cpu" or "cuda"). If None, automatically selected.

        Returns
        -------
        dict
            A dictionary containing model estimates and components:
            - **M** (`torch.Tensor`): estimated covariance matrix of random effects.
            - **s** (`float`): estimated measurement error variance.
            - **negloglik** (`float`): final negative log-likelihood value.
            - **w** (`torch.Tensor`): (K, T) matrix of random-effect estimates per time step.
            - **V** (`torch.Tensor`): (K, K) covariance matrix of prediction errors for `w[t]`.
            - **G** (`dict`): basis function matrix used in fitting.
            - **LKobj** (`dict`): results from LatticeKrig-style fine-scale modeling (if enabled).
            - **tps_method** (`str`): specifies the method used to compute thin-plate splines (TPS).
            - **requires_grad** (`bool`): whether gradient computation was enabled.
            - **dtype** (`torch.dtype`): data type used in computations.
            - **device** (`torch.device`): computation device used.
        """
        # setup device
        if device is None:
            device = setup_device(device = self.device,
                                  logger = True
                                  )
        else:
            device = setup_device(device = device,
                                  logger = True
                                  )
            self.device = device

        # dtype check
        if dtype is None:
            if self.dtype is not None:
                dtype = self.dtype
            elif isinstance(data, torch.Tensor):
                dtype = data.dtype
            else:
                warn_msg = f"Parameter \"dtype\" was not set, Please input a `torch.dtype` instance or a Tensor with dtype. Use default `torch.float64`."
                LOGGER.warning(warn_msg)
                dtype = torch.float64
        elif not isinstance(dtype, torch.dtype):
            warn_msg = f"Invalid dtype: expected a `torch.dtype` instance, got `{type(dtype).__name__}`, use default `torch.float64`."
            LOGGER.warning(warn_msg)
            dtype = torch.float64
        self.dtype = dtype

        # method check
        if method not in ["fast", "EM"]:
            error_msg = f'The specified method "{method}" is not supported. Available methods are "fast" and "EM".'
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

        # convert all major parameters
        mu = to_tensor(mu, dtype=dtype, device=device)
        if mu.ndim == 0:
            pass
        elif mu.ndim == 1:
            mu = mu.unsqueeze(1)
        elif mu.ndim == 2 and mu.shape[1] == 1:
            pass
        else:
            error_msg = f'Invalid shape for "mu": expected scalar or (n,) tensor, got {mu.shape}'
            LOGGER.error(error_msg)
            raise ValueError(error_msg)
        if mu.ndim != 0 and data.shape[0] != mu.shape[0]:
            error_msg = f'Shape mismatch between "data" and "mu": data has {data.shape[0]} rows, mu has {mu.shape[0]} rows'
            LOGGER.error(error_msg)
            raise ValueError(error_msg)
        D = to_tensor(D, dtype=dtype, device=device) if D is not None else None
        G = to_tensor(G, dtype=dtype, device=device) if G is not None else None
        Kseq = to_tensor(Kseq, dtype=dtype, device=device) if Kseq is not None else None

        # convert data and locations
        data = to_tensor(data, dtype=dtype, device=device)
        loc = to_tensor(loc, dtype=dtype, device=device)

        # reshape data
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if loc.ndim == 1:
            loc = loc.reshape(-1, 1)

        # check data and loc shape
        if data.shape[0] != loc.shape[0]:
            error_msg = f'Shape mismatch between "data" and "loc": data has {data.shape[0]} rows, loc has {loc.shape[0]} rows'
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

        # check tps_method
        if not isinstance(tps_method, str):
            tps_method = int(tps_method)
        if tps_method == 0 or tps_method == "rectangular":
            tps_method = "rectangular"
        elif tps_method == 1 or tps_method == "spherical":
            tps_method = "spherical"
        elif tps_method == 2 or tps_method == "spherical_fast":
            tps_method = "spherical_fast"
        else:
            warn_msg = f'Invalid tps_method "{tps_method}", it should be one of "rectangular", "spherical_fast", or "spherical", using default "rectangular" method instead.'
            LOGGER.warning(warn_msg)
        LOGGER.info(f'Calculate TPS with {tps_method}.')
        self.tps_method = tps_method

        # requires_grad check
        if requires_grad:
            data.requires_grad_(requires_grad = True)
            info_msg = f"Gradient tracking has been enabled for autoFRK."
            LOGGER.info(info_msg)
            # methods quited to use
            #if method in ["fast_sklearn", "fast_faiss"]:
            #    warn_msg = f' Gradient tracking can only suppose methods "fast" and "EM", now switch to "fast".'
            #    LOGGER.warning(warn_msg)
            #    method = "fast"

        # run code by requires_grad
        with torch.set_grad_enabled(mode = requires_grad):
            data = data - mu
            Fk = {}
            if G is not None:
                Fk["MRTS"] = G
            else:
                Fk = selectBasis(data           = data, 
                                 loc            = loc,
                                 D              = D, 
                                 maxit          = maxit, 
                                 avgtol         = tolerance,
                                 max_rank       = maxK, 
                                 sequence_rank  = Kseq, 
                                 method         = method, 
                                 num_neighbors  = n_neighbor,
                                 max_knot       = maxknot, 
                                 DfromLK        = None,
                                 Fk             = None,
                                 tps_method     = tps_method,
                                 dtype          = dtype,
                                 device         = device
                                 )
            
            K = Fk["MRTS"].shape[1]
            if method == "fast":
                data = fast_mode_knn_torch(data       = data,
                                           loc        = loc, 
                                           n_neighbor = n_neighbor
                                           )
                
            if not finescale:
                obj = indeMLE(data      = data,
                              Fk        = Fk["MRTS"][:, :K],
                              D         = D,
                              maxit     = maxit,
                              avgtol    = tolerance,
                              wSave     = True,
                              DfromLK   = None,
                              vfixed    = None,
                              verbose   = True,
                              dtype     = dtype,
                              device    = device
                              )
                
            else:
                """
                In the R package `autoFRK`, this functionality is implemented using the `LatticeKrig` package.
                This implementation is not provided in the current context.
                """
                error_msg = "The part about \"method == else\" in `AutoFRK.forward()` is Not provided yet!"
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
                                          weights=1.0 / torch.diag(D),
                                          n_neighbor=n_neighbor,
                                          nu=nu
                                          )
                
                DnLK = setLKnFRKOption(LK_obj,
                                       Fk["MRTS"][:, :K],
                                       nc=NC,
                                       a_wght=a_wght
                                       )
                DfromLK = DnLK['DfromLK']
                LKobj = DnLK['LKobj']
                obj = indeMLE(data=data,
                              Fk=Fk["MRTS"][:, :K],
                              D=D,
                              maxit=maxit,
                              avgtol=tolerance,
                              wSave=True,
                              DfromLK=DfromLK,
                              vfixed=DnLK.get('s', None)
                              )
            
        obj['G'] = Fk
        obj['tps_method'] = self.tps_method
        
        if finescale:
            """
            In the R package `autoFRK`, this functionality is implemented using the `LatticeKrig` package.
            This implementation is not provided in the current context.
            """
            error_msg = "The part about \"if finescale\" in `AutoFRK.forward()` is Not provided yet!"
            LOGGER.error(error_msg)
            raise NotImplementedError(error_msg)
        
            obj['LKobj'] = LKobj
            obj.setdefault('pinfo', {})
            obj['pinfo']["loc"] = loc
            obj['pinfo']["weights"] = 1.0 / torch.diag(D)
        else:
            obj['LKobj'] = None
        
        obj['requires_grad'] = requires_grad
        obj['dtype'] = dtype
        obj['device'] = device
        self.obj = obj

        garbage_cleaner()
        return self.obj
    
    def predict(
        self,
        obj: dict = None,
        obsData: torch.Tensor = None,
        obsloc: torch.Tensor = None,
        mu_obs: Union[float, torch.Tensor] = 0,
        newloc: torch.Tensor = None,
        basis: torch.Tensor = None,
        mu_new: Union[float, torch.Tensor] = 0,
        se_report: bool = False,
        tps_method: str | int | None = None,
        dtype: torch.dtype | None = None,
        device: Optional[Union[torch.device, str]]=None
    ) -> dict:
        """
        `autoFRK` predict method

        Predict values and (optionally) standard errors from a fitted autoFRK model.

        Parameters
        ----------
        obj : dict, optional
            Model object returned by `forward()`. If None, uses `self.obj`.
        obsData : torch.Tensor, optional
            Observed data used for prediction. Default uses data stored in `obj`.
        obsloc : torch.Tensor, optional
            Coordinates of observation locations corresponding to `obsData`.
            Only applicable if `obj['G']` uses automatically generated TPS basis functions.
            Default uses the locations stored in `obj`.
        mu_obs : float or torch.Tensor, optional
            Deterministic mean values at `obsloc`. Default is 0.
        newloc : torch.Tensor, optional
            Coordinates of new locations for prediction. Default is `None`, 
            which predicts at the observation locations.
        basis : torch.Tensor, optional
            Basis function matrix evaluated at `newloc`. Each column is a basis function.
            Can be omitted if `obj` used automatically generated TPS basis functions.
        mu_new : float or torch.Tensor, optional
            Deterministic mean values at `newloc`. Default is 0.
        se_report : bool, optional
            If True, standard errors of the predictions are returned. Default is False.
        tps_method : str, int or None, optional
            Specifies the method used to compute thin-plate splines (TPS). Default is None.
            Options:
                - None: Auto detect by `forward` method.
                - "rectangular" (or 0): Compute TPS in Euclidean (rectangular) coordinates.
                - "spherical" (or 1): Compute TPS directly in spherical coordinates.
                - "spherical_fast" (or 2): Use spherical coordinates but apply the rectangular TPS formulation for faster computation.
        dtype : torch.dtype or None
            The data type for computations. If different from the object's dtype, tensors will be converted. Default is None (auto detected).
        device : torch.device or str, optional
            The device on which computations will be performed (CPU or GPU). 
            If None, will use the device stored in `obj` or `self.device`.

        Returns
        -------
        dict
            Dictionary containing prediction results:
            - **pred.value** (`torch.Tensor`): predicted values at the new locations.
            - **se** (`torch.Tensor`, optional): standard errors of predictions (if `se_report=True`).
        """
        if obj is None and not hasattr(self, "obj"):
            error_msg = f'No input "obj" is provided and `AutoFRK.forward` has not been called before `AutoFRK.predict`.'
            LOGGER.error(error_msg)
            raise ValueError(error_msg)
        elif obj is None and hasattr(self, "obj"):
            obj = self.obj

        # setup object type
        change_tensor = False

        # setup device
        obj['device'] = obj.get('device', None)
        if device is None:
            if obj['device'] is not None:
                device = obj['device']
            else:
                device = self.device
        elif device == obj['device']:
            device = obj['device']
        elif device == self.device:
            device = self.device
        else:
            device = setup_device(device = device,
                                  logger = True
                                  )
            change_tensor = True
        self.device = device

        # check dtype
        obj['dtype'] = obj.get('dtype', None)
        if dtype is None:
            if obj['dtype'] is not None and isinstance(obj['dtype'], torch.dtype):
                dtype = obj['dtype']
            elif self.dtype is not None and isinstance(self.dtype, torch.dtype):
                dtype = self.dtype
            else:
                warn_msg = f"Parameter \"dtype\" was not set, Please input a `torch.dtype` instance or a Tensor with dtype. Use default `torch.float64`."
                LOGGER.warning(warn_msg)
                dtype = torch.float64
        elif dtype == obj['dtype'] or dtype == self.dtype:
            pass
        elif not isinstance(dtype, torch.dtype):
            warn_msg = f"Invalid dtype: expected a `torch.dtype` instance, got `{type(dtype).__name__}`, use default `torch.float64`."
            LOGGER.warning(warn_msg)
            dtype = torch.float64
        else:
            change_tensor = True
        self.dtype = dtype

        # convert all major parameters
        if change_tensor:
            obj = to_tensor(obj     = obj,
                            dtype   = self.dtype,
                            device  = self.device
                            )
        obsData = to_tensor(obsData, dtype=dtype, device=device) if obsData is not None else None
        obsloc = to_tensor(obsloc, dtype=dtype, device=device) if obsloc is not None else None
        mu_obs = to_tensor(mu_obs, dtype=dtype, device=device)
        newloc = to_tensor(newloc, dtype=dtype, device=device) if newloc is not None else None
        basis = to_tensor(basis, dtype=dtype, device=device) if basis is not None else None
        mu_new = to_tensor(mu_new, dtype=dtype, device=device)
        if mu_new.ndim == 0:
            pass
        elif mu_new.ndim == 1:
            mu_new = mu_new.unsqueeze(1)
        elif mu_new.ndim == 2 and mu_new.shape[1] == 1:
            pass
        else:
            error_msg = f'Invalid shape for "mu_new": expected scalar or (n,) tensor, got {mu_new.shape}'
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

        # check tps_method
        if tps_method is None:
            if obj.get('tps_method', None) is not None:
                tps_method = obj['tps_method']
            elif hasattr(self, "tps_method"):
                warn_msg = f'No input "tps_method" is provided, use the default value `"rectangular"` for MRTS.predict.'
                LOGGER.warning(warn_msg)
                tps_method = "rectangular"
            else:
                error_msg = f'Could not find the parameter "tps_method". Please specify a valid method ("rectangular", "spherical_fast" or "spherical").'
                LOGGER.error(error_msg)
                ValueError(error_msg)
        if not isinstance(tps_method, str):
            tps_method = int(tps_method)
        if tps_method == 0 or tps_method == "rectangular":
            tps_method = "rectangular"
        elif tps_method == 1 or tps_method == "spherical":
            tps_method = "spherical"
        elif tps_method == 2 or tps_method == "spherical_fast":
            tps_method = "spherical_fast"
        else:
            error_msg = f'Invalid tps_method "{tps_method}", it should be one of "rectangular", "spherical_fast", or "spherical".'
            LOGGER.error(error_msg)
            ValueError(error_msg)
        self.tps_method = tps_method
            
        return predict_FRK(obj          = obj,
                           obsData      = obsData,
                           obsloc       = obsloc,
                           mu_obs       = mu_obs,
                           newloc       = newloc,
                           basis        = basis,
                           mu_new       = mu_new,
                           se_report    = se_report,
                           tps_method   = tps_method,
                           dtype        = self.dtype,
                           device       = self.device
                           )

# main program
if __name__ == "__main__":
    print("This is the class `AutoFRK` for autoFRK package. Please import it in your code to use its functionalities.")

"""
Title: Multi-Resolution Thin-plate Spline basis function for Spatial Data, and calculate the basis function by using rectangular or spherical coordinates.
Author: Hsu, Yao-Chih
Version: 1140611
Reference: Resolution Adaptive Fixed Rank Kringing by ShengLi Tzeng & Hsin-Cheng Huang
"""

# import modules
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import quad
from typing import Optional, Union, Any
from autoFRK.utils.logger import setup_logger
from autoFRK.utils.device import setup_device
from autoFRK.utils.utils import to_tensor

# logger config
LOGGER = setup_logger()

# classes
class MRTS(nn.Module):
    def __init__(
        self,
        locs: torch.Tensor, 
        k: int=None, 
        calculate_with_spherical: bool=False,
        standardize: bool=True,
        unbiased_std: bool=False,
        dtype: torch.dtype=torch.float64,
        device: Union[torch.device, str]='cpu'
    ):
        """
        Multi-Resolution Thin-plate Spline basis function.
        
        Args:
            calculate_with_spherical: (Optional) If True, use spherical coordinates for TPS calculation (default is False) (only supported when d == 2).
            standardize: (Optional) If True, perform standardization on locs (Recommended).
            unbiased_std: (Optional) If True, use unbiased standard deviation (dividing by N-1) (Recommended); 
                          if False, use biased standard deviation (dividing by N).
        """
        super().__init__()
        self.dtype = dtype
        self.device = setup_device(device=device)

        locs = to_tensor(obj   = locs,
                         dtype = dtype,
                         device= device
                         )
        if locs.ndim == 1:
            locs = locs.unsqueeze(1)
        self.locs = locs

        self.N, self.d = self.locs.shape

        # number of basis
        max_k = self.N + self.d
        try:
            if k is None:
                self.k = max_k
                LOGGER.warning(f'Parameter "k" was not set. Default value {self.k} is used.')
            elif 0 < k <= (max_k):
                self.k = k
            else:
                self.k = max_k
                LOGGER.warning(f'Parameter "k" is out of valid range, it should be between 1 to {self.k}. Default value {self.k} is used.')
        except TypeError:
            self.k = max_k
            LOGGER.warning(f'Parameter "k" is not an integer, the type you input is "{type(k).__name__}." Default value {self.k} is used.')

        self.calculate_with_spherical = calculate_with_spherical
        if type(self.calculate_with_spherical) is not bool:
            self.calculate_with_spherical = False
            LOGGER.warning(f'Parameter "calculate_with_spherical" should be a boolean, the type you input is "{type(calculate_with_spherical).__name__}". Default value False is used.')
        elif self.calculate_with_spherical and self.d != 2:
            self.calculate_with_spherical = False
            LOGGER.warning(f'Spherical TPS not implemented for d = {self.d}, only d = 2 is supported. Using rectangular coordinate system instead.')
        
        if self.calculate_with_spherical:
            LOGGER.info(f'Calculate TPS with spherical coordinates.')
        else:
            LOGGER.info(f'Calculate TPS with rectangular coordinates.')

        self.standardize = standardize
        if type(self.standardize) is not bool:
            self.standardize = True
            LOGGER.warning(f'Parameter "standardize" should be a boolean, the type you input is "{type(standardize).__name__}". Default value True is used.')

        self.unbiased_std = unbiased_std
        if type(self.unbiased_std) is not bool:
            self.unbiased_std = False
            LOGGER.warning(f'Parameter "unbiased_std" should be a boolean, the type you input is "{type(unbiased_std).__name__}". Default value False is used.')

    def _tps_phi(
        self,
        locs: torch.Tensor,
        calculate_with_spherical: bool=False,
        dtype: torch.dtype = torch.float64,
        device: Union[torch.device, str]='cpu'
    ) -> torch.Tensor:
        """
        Compute the Thin Plate Spline (TPS) radial basis matrix for input locations.

        Args:
            locs: [N, d] tensor of spatial locations.
            calculate_with_spherical: (Optional) If True, use spherical coordinates for TPS calculation (default is False) (only supported when d == 2).

        Returns:
            A [N, N] tensor representing the TPS radial basis matrix.
        """       

        # rectangular coordinate system
        if not calculate_with_spherical:
            return self.__calculate_phi_by_rectangular_coordinate(locs = locs)
        
        # spherical coordinate system
        else:
            # convert to spherical coordinates
            ret = torch.zeros((self.N, self.N), dtype=dtype, device=device)
            locs = locs * torch.pi / 180.0
            x = locs[:, 0]
            y = locs[:, 1]
            for i in range(self.N):
                for j in range(self.N):
                    ret[i, j] = self.__calculate_phi_by_spherical_coordinate(x1     = x[i],
                                                                             y1     = y[i],
                                                                             x2     = x[j],
                                                                             y2     = y[j],
                                                                             dtype  = dtype,
                                                                             device = device
                                                                             )
            return ret

    def __calculate_phi_by_rectangular_coordinate(
        self,
        locs: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the TPS radial basis matrix using rectangular (Euclidean) coordinates.

        Args:
            locs: [N, d] tensor of spatial locations.

        Returns:
            A [N, N] tensor representing the TPS radial basis matrix.
        """
        dists = torch.cdist(locs, locs)  # Euclidean distances
        if self.d == 1:
            return (dists ** 3) / 12
        
        elif self.d == 2:
            mask = dists != 0
            ret = torch.zeros_like(dists)
            ret[mask] = (dists[mask] ** 2) * torch.log(dists[mask]) / (8 * torch.pi)
            return ret
        
        elif self.d == 3:
            return -dists / 8
        
        else:
            error_msg = f"TPS not implemented for d = {self.d}"
            LOGGER.error(error_msg)
            raise NotImplementedError(error_msg)

    def __calculate_phi_by_spherical_coordinate(
        self,
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor,
        dtype: torch.dtype = torch.float64,
        device: Union[torch.device, str]='cpu'
    ) -> torch.Tensor:
        """
        Calculate the TPS radial basis function using spherical coordinates.
        
        Args:
            x1, y1: Spherical coordinates of the first point (in radians).
            x2, y2: Spherical coordinates of the second point (in radians). 

        Returns:
            A float representing the TPS radial basis function value.
        """
        cos_theta = torch.sin(x1) * torch.sin(x2) + torch.cos(x1) * torch.cos(x2) * torch.cos(y1 - y2)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        ret = 1 - (torch.pi ** 2) / 6

        if not torch.isclose(cos_theta, torch.tensor(-1.0, dtype=dtype, device=device)):
            cos_theta = cos_theta.item()
            upper = (cos_theta + 1) / 2
            lower = 0
            result, _ = quad(func   = self.__integrand_for_spherical_in_tps,
                             a      = lower,
                             b      = upper,
                             epsabs = 1e-6,
                             epsrel = 1e-6,
                             limit  = 50
                             )
            result = to_tensor(result, dtype=dtype, device=device)
            ret -= result
        return ret
    
    def __integrand_for_spherical_in_tps(
        self,
        x: Any
    ) -> float:
        """
        Integrand function for the spherical TPS radial basis function.

        Args:
            x: The variable of integration.

        Returns:
            The value of the integrand at x.
        """
        return np.log(1 - x) / x
        
    def _standardize(
        self,
        x: torch.Tensor,
        unbiased_std: Optional[bool] = None
    ) -> torch.Tensor:
        """
        A function to standardize the input tensor x.

        Args:
            x: Input tensor to be standardized.
            unbiased_std: (Optional) If True, use unbiased standard deviation (dividing by N-1); 
                          if False, use biased standard deviation (dividing by N).
                          Default is None, which uses the class attribute self.unbiased_std.

        Returns:
            A standardized tensor with mean 0 and standard deviation 1.
        """
        if unbiased_std is None:
            unbiased_std = self.unbiased_std
        mean = x.mean(dim=0, keepdim=True)  # keepdim=True to keep the same shape
        std = x.std(dim=0, unbiased=unbiased_std, keepdim=True)  # unbiased=False for population std
        std[std == 0] = 1.0
        x = (x - mean) / std
        return x

    def forward(
        self
    ) -> torch.Tensor:
        """
        Forward pass to compute the basis functions. Constructs the basis functions based on the input locations and parameters.

        Returns:
            A tensor of shape [N, k] representing the basis functions.
        """
        # initialize
        dtype = self.dtype
        device = self.device
        
        # Construct X = [1, x1, x2, ..., xd]
        ones = torch.ones(self.N, 1, dtype=dtype, device=device)
        X = torch.cat([ones, self.locs], dim=1)

        if self.k == 1:
            return ones
        elif self.k <= self.d:
            if self.standardize:
                X[:, 1:self.k] = self._standardize(x            = X[:, 1:self.k],
                                                   unbiased_std = None
                                                   )
            return X[:, :self.k]
        

        # TPS basis
        Phi = self._tps_phi(locs                    = self.locs,
                            calculate_with_spherical= self.calculate_with_spherical,
                            dtype                   = dtype,
                            device                  = device
                            )
        
        try:
            XtX_inv = torch.inverse(X.T @ X)
        except RuntimeError as e:
            LOGGER.warning(f'torch.inverse failed due to {e}, this implies "X.T @ X" didn\'t have inverse. Using torch.linalg.pinv instead.')
            XtX_inv = torch.linalg.pinv(X.T @ X)
        Q = torch.eye(self.N, dtype=dtype, device=device) - X @ XtX_inv @ X.T

        # Eigen-decomposition
        G = Q @ Phi @ Q
        eigenvalues, eigenvectors = torch.linalg.eigh(G)  # ascending order
        idx_desc = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx_desc]
        eigenvectors = eigenvectors[:, idx_desc]

        Basis_high = eigenvectors * torch.sqrt(torch.tensor(self.N, dtype=dtype, device=device))
        Basis_low = X

        # Standardize
        if self.standardize:
            Basis_low[:, 1:(self.d + 1)] = self._standardize(x              = Basis_low[:, 1:(self.d + 1)],
                                                             unbiased_std   = None
                                                             )

        F = torch.cat([Basis_low, Basis_high], dim=1)

        # Output k basis
        F = F[:, :self.k]

        return F


# main program
if __name__ == "__main__":
    print("This is the class `MRTS` for autoFRK package. Please import it in your code to use its functionalities.")









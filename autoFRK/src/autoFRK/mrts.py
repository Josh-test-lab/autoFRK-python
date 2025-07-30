"""
Title: Multi-Resolution Thin-plate Spline basis function for Spatial Data, and calculate the basis function by using rectangular or spherical coordinates.
Author: Hsu, Yao-Chih
Version: 1140611
Reference: Resolution Adaptive Fixed Rank Kringing by ShengLi Tzeng & Hsin-Cheng Huang
"""

# import modules
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import quad
from scipy.sparse.linalg import eigsh
from typing import Optional, Union
import datetime
from autoFRK.utils.logger import setup_logger
from autoFRK.utils.device import setup_device

# logger config
LOGGER = setup_logger()

# classes
class MRTS(nn.Module):
    def __init__(self,
                 locs: torch.Tensor, 
                 k: int=None, 
                 device: Optional[Union[torch.device, str]]=None, 
                 calculate_with_spherical: bool=False, 
                 standardize: bool=True, 
                 unbiased_std: bool=False
                 ):
        """
        Multi-Resolution Thin-plate Spline basis function.
        
        Args:
            locs: [N, d] tensor of spatial locations.
            k: (Optional) parameter for controlling resolution.
            device: (Optional) PyTorch device to use.
            calculate_with_spherical: (Optional) If True, use spherical coordinates for TPS calculation (default is False) (only supported when d == 2).
            standardize: (Optional) If True, perform standardization on locs (Recommended).
            unbiased_std: (Optional) If True, use unbiased standard deviation (dividing by N-1) (Recommended); 
                          if False, use biased standard deviation (dividing by N).
        """
        super().__init__()
        self.device = setup_device(device=device)

        self.register_buffer("locs", locs.to(self.device))
        self.N, self.d = locs.shape

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

    def _tps_phi(self, locs: torch.Tensor, calculate_with_spherical: bool=False) -> torch.Tensor:
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
            return self.__calculate_phi_by_rectangular_coordinate(locs)
        
        # spherical coordinate system
        else:
            # convert to spherical coordinates
            ret = torch.zeros((self.N, self.N), device=locs.device)
            locs = locs * torch.pi / 180.0
            x = locs[:, 0]
            y = locs[:, 1]
            for i in range(self.N):
                for j in range(self.N):
                    ret[i, j] = self.__calculate_phi_by_spherical_coordinate(x[i], y[i], x[j], y[j])
            return ret

    def __calculate_phi_by_rectangular_coordinate(self, locs: torch.Tensor) -> torch.Tensor:
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
            raise NotImplementedError(f"TPS not implemented for d = {self.d}")

    def __calculate_phi_by_spherical_coordinate(self, x1: float, y1: float, x2: float, y2: float) -> float:
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
        #theta = torch.acos(cos_theta)
        ret = 1 - (torch.pi ** 2) / 6

        if not torch.isclose(cos_theta, torch.tensor(-1.0, device=cos_theta.device)):
            cos_theta = cos_theta.item()
            upper = (cos_theta + 1) / 2
            lower = 0
            result, _ = quad(self.__integrand_for_spherical_in_tps, lower, upper, epsabs=1e-6, epsrel=1e-6, limit=50)
            result = torch.tensor(result, dtype=torch.float32, device=self.device)
            ret -= result
        return ret
    
        # # method by GPT can just use tenser to calculate and efficient
        # # spherical coordinate system
        # locs_rad = locs * torch.pi / 180.0  # convert to radians
        # lat1 = locs_rad[:, 0].unsqueeze(1)  # [N, 1]
        # lon1 = locs_rad[:, 1].unsqueeze(1)  # [N, 1]
        # lat2 = locs_rad[:, 0].unsqueeze(0)  # [1, N]
        # lon2 = locs_rad[:, 1].unsqueeze(0)  # [1, N]

        # # Cosine of angular distance
        # cos_theta = (
        #     torch.sin(lat1) @ torch.sin(lat2) + torch.cos(lat1) @ torch.cos(lat2) * torch.cos(lon1 - lon2)
        # )
        # cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        # # Compute integral using trapezoidal rule
        # N = 100  # Number of steps in the integral approximation
        # x = torch.linspace(1e-6, 1.0, N, device=locs.device)  # avoid x=0
        # ln_term = torch.log(1 - x) / x  # [N]

        # def trapezoid_integrate(upper):  # upper: [N, N]
        #     upper = torch.clamp(upper, 1e-6, 1.0)
        #     area = torch.zeros_like(upper)
        #     for i in range(N - 1):
        #         x0, x1 = x[i], x[i + 1]
        #         y0, y1 = ln_term[i], ln_term[i + 1]
        #         h = x1 - x0
        #         area += h * (y0 + y1) / 2 * ((upper >= x1).float())  # only add where upper > x1
        #     return area

        # upper_bound = (cos_theta + 1) / 2  # [N, N]
        # integral_vals = trapezoid_integrate(upper_bound)  # [N, N]
        # ret = 1 - (torch.pi ** 2) / 6 - integral_vals
        # return ret
    
    def __integrand_for_spherical_in_tps(self, x):
        """
        Integrand function for the spherical TPS radial basis function.

        Args:
            x: The variable of integration.

        Returns:
            The value of the integrand at x.
        """
        return np.log(1 - x) / x
        
    def _standardize(self, x: torch.Tensor, unbiased_std: Optional[bool] = None) -> torch.Tensor:
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

    def forward(self) -> torch.Tensor:
        """
        Forward pass to compute the basis functions. Constructs the basis functions based on the input locations and parameters.

        Returns:
            A tensor of shape [N, k] representing the basis functions.
        """

        # Construct X = [1, x1, x2, ..., xd]
        ones = torch.ones(self.N, 1, device=self.device)
        X = torch.cat([ones, self.locs], dim=1)  # [N, d+1]

        if self.k == 1:
            return ones
        elif self.k <= self.d:
            if self.standardize:
                X[:, 1:self.k] = self._standardize(X[:, 1:self.k])
            return X[:, :self.k]
        

        # TPS basis
        Phi = self._tps_phi(locs=self.locs, calculate_with_spherical=self.calculate_with_spherical)  # [N, N]
        
        # Q = I - X(XᵗX)⁻¹Xᵗ
        try:
            XtX_inv = torch.inverse(X.T @ X)  # [(d+1), (d+1)]
        except RuntimeError as e:
            LOGGER.warning(f'torch.inverse failed due to {e}, this implies "X.T @ X" didn\'t have inverse. Using torch.linalg.pinv instead.')
            XtX_inv = torch.linalg.pinv(X.T @ X)
        Q = torch.eye(self.N, device=self.device) - X @ XtX_inv @ X.T

        # Eigen-decomposition
        G = Q @ Phi @ Q
        eigenvalues, eigenvectors = torch.linalg.eigh(G)  # ascending order
        
        # G = G.detach().cpu().numpy()
        # eigenvalues, eigenvectors = eigsh(G, k=self.k - self.d - 1, which='LM')  # 'LM': Largest Magnitude, max k < N
        # eigenvalues = torch.from_numpy(eigenvalues).to(self.device) 
        # eigenvectors = torch.from_numpy(eigenvectors).to(self.device)

        # Filter out near-zero eigenvalues and sort descending
        #valid = eigenvalues > 1e-10
        #eigenvalues[~valid] *= -1.0  # make them positive
        #eigenvectors[:, ~valid] *= -1.0  # make them positive
        idx_desc = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx_desc]
        eigenvectors = eigenvectors[:, idx_desc]

        Basis_high = eigenvectors * torch.sqrt(torch.tensor(self.N, dtype=torch.float32))

        # # Construct basis functions
        # Basis_high = (Phi.T - Phi @ X @ XtX_inv @ X.T).T @ eigenvectors @ torch.diag(1.0 / eigenvalues).to(self.device)
        
        # # the for loop
        # Basis_high_1 = torch.zeros(self.N, self.N, device=self.locs.device)
        # for i in range(self.N):
        #     phi_i = Phi[i, :]
        #     x_i = X[i, :]
        #     Phi_XXtXinv_x = (phi_i.T - Phi @ X @ XtX_inv @ x_i.T).T
        #     for j in range(self.N):
        #         lambda_j = 1 / eigenvalues[j]
        #         v_j = eigenvectors[:, j]
        #         Basis_high_1[i, j] = lambda_j * Phi_XXtXinv_x @ v_j
        # print(f'Basis_high:\n{Basis_high}, \n\nBasis_high_1:\n{Basis_high_1}')
        # print(torch.mean((Basis_high - Basis_high_1)**2))

        # Concatenate [1, x1, ..., xd] and B_high

        Basis_low = X  # [N, d+1]

        # Standardize
        if self.standardize:
            Basis_low[:, 1:(self.d + 1)] = self._standardize(Basis_low[:, 1:(self.d + 1)])

        F = torch.cat([Basis_low, Basis_high], dim=1)

        # Output k basis
        F = F[:, :self.k]

        return F  # [N, d+1 + k]
    
# functions


# main program
if __name__ == "__main__":
    # time
    start_time = datetime.datetime.now()

    # Load locations
    locations = np.load(f'locations.npy')

    print(f'locations:\n{locations}')
    print(f'locations shape: {locations.shape}')

    # Convert to tensor
    locs = torch.tensor(locations, dtype=torch.float32)

    # Create FRK basis
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frk_basis = MRTS(locs, device=device, calculate_with_spherical=False)

    # Forward pass
    #F = pd.DataFrame(frk_basis().detach().cpu().numpy())
    F = pd.DataFrame(frk_basis())
    #.detach().cpu().numpy()

    print(f'F:\n{F}')
    # print(f'F sum:\n{np.sum(F**2, axis=0)}')
    # print(f'F shape: {F.shape}')

    # time
    end_time = datetime.datetime.now()
    use_time = end_time - start_time
    print(f'Use time: {use_time}')







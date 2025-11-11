"""
Title: Matrix operator utilities for autoFRK-Python Project
Author: Yao-Chih Hsu
Version: 1141026
Description: This file provides functions for matrix operations used in the autoFRK-Python Project.
Reference: `autoFRK` R package by Wen-Ting Wang from https://github.com/egpivo/autoFRK
"""

# import modules
import torch
from typing import Dict, Union, Tuple
from scipy.sparse.linalg import eigsh
from ..utils.utils import to_tensor

# using in predict_FRK
# check = none

def to_sparse(
    tensor: torch.Tensor
) -> torch.Tensor:
    """
    Convert a dense tensor to sparse COO tensor if it is not already sparse.
    Supports any shape and preserves requires_grad, dtype, and device.

    Parameters
    ----------
    tensor : torch.Tensor
        Input dense or sparse tensor.

    Returns
    -------
    torch.Tensor
        Sparse COO tensor if input was dense, otherwise the original sparse tensor.
    """
    if tensor.is_sparse:
        return tensor

    indices = (tensor != 0).nonzero(as_tuple=True)
    values = tensor[indices]
    values.requires_grad = tensor.requires_grad

    sparse_tensor = torch.sparse_coo_tensor(
        indices=torch.stack(indices),
        values=values,
        size=tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device
    )

    return sparse_tensor

# using in selectBasis, computeProjectionMatrix
# check = ok
def getInverseSquareRootMatrix(
    left_matrix: torch.Tensor,
    right_matrix: torch.Tensor,
    tol: float = 1e-10
) -> torch.Tensor:
    """
    Compute the inverse square root of (left_matrix.T @ right_matrix), assuming the product is symmetric.

    Parameters
    ----------
    left_matrix : torch.Tensor
        Tensor of shape (n, k).
    right_matrix : torch.Tensor
        Tensor of shape (n, k).
    tol : float, optional
        Absolute tolerance for comparing diagonal elements (default: 1e-10).

    Returns
    -------
    torch.Tensor
        Inverse square root matrix of shape (k, k).
    """
    mat = left_matrix.T @ right_matrix  # A^T * B
    mat = (mat + mat.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(mat)
    inv_sqrt_eigvals = torch.diag(torch.clamp(eigvals, min=tol).rsqrt())
    return eigvecs @ inv_sqrt_eigvals @ eigvecs.T

# using in cMLEimat
# check = ok
def computeProjectionMatrix(
    Fk1: torch.Tensor, 
    Fk2: torch.Tensor, 
    data: torch.Tensor, 
    S: torch.Tensor=None,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> Dict[str, torch.Tensor]:
    """
    Compute the projection matrix for maximum likelihood estimation.

    Parameters
    ----------
    Fk1 : torch.Tensor
        Matrix of shape (n, K).
    Fk2 : torch.Tensor
        Matrix of shape (n, K).
    data : torch.Tensor
        Matrix of shape (n, T).
    S : torch.Tensor or None, optional
        Matrix of shape (n, n) if provided, else None.
    dtype : torch.dtype, optional
        Desired torch dtype for computation (default: torch.float64).
    device : torch.device or str, optional
        Target device for computation (default: 'cpu').

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with the following keys:
        - 'inverse_square_root_matrix': torch.Tensor of shape (K, K)
        - 'matrix_JSJ': torch.Tensor of shape (K, K)
    """
    if S is not None:
        S = to_tensor(S, dtype=dtype, device=device)

    num_columns = data.shape[1]
    inverse_square_root_matrix = getInverseSquareRootMatrix(left_matrix = Fk1, 
                                                            right_matrix = Fk2
                                                            )
    inverse_square_root_on_Fk2 = inverse_square_root_matrix @ Fk2.T

    if S is None:
        matrix_JSJ = (inverse_square_root_on_Fk2 @ data) @ (inverse_square_root_on_Fk2 @ data).T / num_columns
    else:
        matrix_JSJ = (inverse_square_root_on_Fk2 @ S) @ inverse_square_root_on_Fk2.T

    matrix_JSJ = (matrix_JSJ + matrix_JSJ.T) / 2

    return {"inverse_square_root_matrix": inverse_square_root_matrix,
            "matrix_JSJ": matrix_JSJ
            }

# using in cMLEimat
# check = ok
def invCz(
    R: torch.Tensor, 
    L: torch.Tensor, 
    z: torch.Tensor, 
    dtype: torch.dtype=torch.float64,
    device: Union[torch.device, str]='cpu'
) -> torch.Tensor:
    """
    Compute the vector (1 x p) given R, L, and z.

    Parameters
    ----------
    R : torch.Tensor
        Matrix of shape (p, p).
    L : torch.Tensor
        Matrix of shape (p, K).
    z : torch.Tensor
        Vector of shape (p,) or row matrix of shape (1, p).
    dtype : torch.dtype, optional
        Desired torch dtype for computation (default: torch.float64).
    device : torch.device or str, optional
        Target device for computation (default: 'cpu').

    Returns
    -------
    torch.Tensor
        Resulting tensor of shape (1, p).
    """
    if z.dim() == 1:
        z = z.unsqueeze(1)

    K = L.shape[1]
    try:
        iR = torch.cholesky_inverse(torch.linalg.cholesky(R))
    except RuntimeError:
        iR = torch.linalg.pinv(R)
    iRZ = iR @ z
    tmp = torch.eye(K, dtype=dtype, device=device) + (L.T @ iR @ L)
    try:
        tmp_inv = torch.cholesky_inverse(torch.linalg.cholesky(tmp))
    except RuntimeError:
        tmp_inv = torch.linalg.pinv(tmp)
    right = L @ tmp_inv @ (L.T @ iRZ)
    result = iRZ - iR @ right

    return result.T

# using in EM0miss
# check = ok
def isDiagonal(
    tensor: torch.Tensor,
    tol: float = 1e-10
) -> bool:
    """
    Check if a 2D numeric tensor is diagonal within a specified tolerance.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to check.
    tol : float, optional
        Absolute tolerance for comparing off-diagonal elements (default: 1e-10).

    Returns
    -------
    bool
        True if the tensor is diagonal (or scalar), False otherwise.
    """
    if tensor.numel() == 1:
        return True

    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        return False

    diag = torch.diag(torch.diagonal(tensor))
    return torch.allclose(tensor, diag, atol=tol)

# using in EM0miss
# check = ok
def convertToPositiveDefinite(
    mat: torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> torch.Tensor:
    """
    Convert a square matrix to a positive-definite matrix.

    Parameters
    ----------
    mat : torch.Tensor
        Input 2D matrix (square, symmetric or not).
    dtype : torch.dtype, optional
        Desired torch dtype for computation (default: torch.float64).
    device : torch.device or str, optional
        Target device for computation (default: 'cpu').

    Returns
    -------
    torch.Tensor
        A positive-definite version of the input matrix.
    """
    # Ensure symmetry
    if not torch.allclose(mat, mat.T, atol=1e-10):
        mat = (mat + mat.T) / 2

    try:
        # Compute eigenvalues only
        eigenvalues = torch.linalg.eigvalsh(mat)
        min_eigenvalue = torch.min(eigenvalues).item()
    except RuntimeError:
        # Fallback in case of numerical error
        mat = (mat + mat.T) / 2
        eigenvalues = torch.linalg.eigvalsh(mat)
        min_eigenvalue = torch.min(eigenvalues).item()

    if min_eigenvalue <= 0:
        adjustment = abs(min_eigenvalue) + 1e-10
        mat = mat + torch.eye(mat.shape[0], dtype=dtype, device=device) * adjustment

    return mat

# using in updateMrtsBasisComponents
# check = none
def decomposeSymmetricMatrix(
    M: torch.Tensor,
    k: int,
    ncv: int = None,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str]='cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the largest k eigenvalues and corresponding eigenvectors of a symmetric matrix.

    Parameters
    ----------
    M : torch.Tensor
        Symmetric matrix of shape (n, n).
    k : int
        Number of largest eigenvalues requested. Must satisfy 1 <= k <= n - 1.
    ncv : int, optional
        Lanczos subspace size for large matrices (used with scipy eigsh, and n > 1000).
    dtype : torch.dtype, optional
        Desired torch dtype for computation (default: torch.float64).
    device : torch.device or str, optional
        Target device for computation (default: 'cpu').

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - First element: torch.Tensor of shape (k,) containing the largest k eigenvalues.
        - Second element: torch.Tensor of shape (n, k) containing the corresponding eigenvectors.
    """
    # n = M.shape[0]
    
    # if n > 1000:
    #     M_np = M.cpu().numpy()
    #     if ncv is None:
    #         ncv = max(2 * k + 1, 20)
    #     lambda_, gamma = eigsh(M_np, k=k, which='LA', ncv=ncv)
    #     lambda_ = torch.from_numpy(lambda_).to(dtype=dtype, device=device)
    #     gamma = torch.from_numpy(gamma).to(dtype=dtype, device=device)
    #     idx = lambda_.argsort(descending=True)
    #     lambda_ = lambda_[idx]
    #     gamma = gamma[:, idx]
    # else:
    #     eigenvalues, eigenvectors = torch.linalg.eigh(M)
    #     lambda_ = eigenvalues[-k:].flip(0)
    #     gamma = eigenvectors[:, -k:].flip(1)
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    lambda_ = eigenvalues[-k:].flip(0)
    gamma = eigenvectors[:, -k:].flip(1)

    return lambda_, gamma

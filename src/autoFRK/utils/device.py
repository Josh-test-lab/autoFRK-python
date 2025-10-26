"""
Title: Setup device of autoFRK-Python Project
Author: Yao-Chih Hsu
Version: 1141025
Description: This file is to set up and check the computation device for the autoFRK-Python project.
Reference: None
"""

# import modules
import gc
import torch
from typing import Optional, Union, Any
from ..utils.logger import LOGGER

# setup device
def setup_device(
    device: Optional[Union[torch.device, str]]=None,
    logger: bool=True
) -> torch.device:
    """
    Set up the computation device.

    If no device is specified, the function automatically selects CUDA if available;
    otherwise, it defaults to CPU. Logs device selection and handles invalid input gracefully.

    Parameters
    ----------
    device : torch.device or str or None, optional
        The computation device. Can be a `torch.device` object or a string such as
        'cpu', 'cuda', or 'cuda:0'. If None, automatically detects available device.
    logger : bool, optional
        Whether to log messages about device selection. Default is True.

    Returns
    -------
    torch.device
        The selected device for computation.
    """
    try:
        if device is None:
            device = detect_device()
            if logger:
                LOGGER.warning(f'Parameter "device" was not set. Value "{device}" detected and used.')
        else:
            device = torch.device(device)
            if logger:
                LOGGER.info(f'Successfully using device "{device}".')
    except (TypeError, ValueError) as e:
        if logger:
            LOGGER.warning(f'Parameter "device" is not a valid device ({device}). Default value "cpu" is used. Error: {e}')
        device = torch.device('cpu')

    return device

# detect device
def detect_device() -> torch.device:
    """
    Automatically select the best available PyTorch device.

    Returns
    -------
    torch.device
        The available device for PyTorch operations.
    """

    # CUDA: NVIDIA GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        return device

    # MPS: Apple Silicon GPU
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps:0")
        return device

    # TPU: via torch_xla
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        return device
    except ImportError:
        pass

    # Default to CPU
    device = torch.device("cpu")

    return device

# check_device
def check_device(
    obj: Any,
    device: Union[torch.device, str]=None
) -> torch.device:
    """
    Automatically determine the torch.device of an input object or nested structure.

    Recursively inspects the input to find a `torch.Tensor` and infer its device.
    Supports arbitrarily nested containers such as dicts, lists, tuples, or sets.
    If a preferred device is provided but differs from the detected one, a warning
    is issued and the detected device is used instead.

    Parameters
    ----------
    obj : Any
        Input object to inspect. Can be a tensor or a container containing tensors.
    device : torch.device, str, or None, optional
        Preferred device. If None, the device is inferred automatically.

    Returns
    -------
    torch.device
        The detected or validated device.
    """
    def _find_device_recursive(obj):
        """
        Recursively search for a tensor device.
        """
        if isinstance(obj, torch.Tensor):
            return obj.device
        elif isinstance(obj, dict):
            for v in obj.values():
                d = _find_device_recursive(v)
                if d is not None:
                    return d
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                d = _find_device_recursive(v)
                if d is not None:
                    return d
        return None

    # find the device from the object
    detected_device = _find_device_recursive(obj)

    if detected_device is None:
        if device is None:
            LOGGER.warning('Could not determine device from input object. Defaulting to CPU.')
            return torch.device('cpu')
        else:
            return torch.device(device)

    if device is not None:
        input_device = torch.device(device)
        if input_device != detected_device:
            warn_msg = f'The input object\'s device ({detected_device}) is different from your input arg "device" ({input_device}); using object\'s device instead.'
            LOGGER.warning(warn_msg)
        return detected_device

    return detected_device

def garbage_cleaner() -> None:
    """
    Universal garbage cleaner for all supported PyTorch devices.
    Performs:
        - Python garbage collection
        - GPU/MPS/XLA cache clearing (if available)
    """
    # Force Python-level garbage collection
    _ = gc.collect()

    # CUDA cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Apple Silicon (MPS) cleanup
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    # TPU cleanup (torch_xla)
    try:
        import torch_xla.core.xla_model as xm
        xm.mark_step()
        xm.wait_device_ops()
    except Exception:
        pass
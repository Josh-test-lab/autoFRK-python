import torch
from typing import Optional, Union, Any
from autoFRK.utils.logger import setup_logger

# logger config
LOGGER = setup_logger()

# setup device
def setup_device(
    device: Optional[Union[torch.device, str]]=None
) -> torch.device:
    """
    Set up the device for computations.

    Args:
        device: The device to use for computations. If None, it defaults to 'cpu'. If a string is provided, it should be a valid device name.
    """
    try:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            LOGGER.warning(f'Parameter "device" was not set. Value "{device}" detected and used.')
        else:
            device = torch.device(device)
            LOGGER.info(f'Successfully using device "{device}".')
    except (TypeError, ValueError) as e:
        LOGGER.warning(f'Parameter "device" is not a valid device ({device}). Default value "cpu" is used. Error: {e}')
        device = torch.device('cpu')

    return device

# check_device
def check_device(
    obj: Any,
    device: Union[torch.device, str]=None
) -> torch.device:
    """
    Automatically determine the torch.device of an input object or nested structure.

    This function recursively inspects the input to locate a torch.Tensor and
    infer the device it resides on. It supports arbitrary nesting, such as dicts,
    lists, tuples, sets, or combinations of these. If a device argument is provided
    but differs from the detected device, a warning is issued, and the detected
    device is used instead.

    Parameters:
        obj (Any): 
            Input object to inspect. Can be a tensor, or container (dict, list, tuple, set)
            containing tensors.
        device (torch.device, str, None, optional): 
            Preferred device. If None, it will be inferred automatically.
    
    Returns:
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

import torch
from typing import Optional, Union
from autoFRK.utils.logger import setup_logger

# logger config
LOGGER = setup_logger()

# setup device
def setup_device(device: Optional[Union[torch.device, str]]=None):
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
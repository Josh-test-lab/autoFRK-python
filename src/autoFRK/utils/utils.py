"""
Title: Useful tools for autoFRK-Python Project
Author: Yao-Chih Hsu
Version: 1141018
Description: This file provides general-purpose utility functions.
Reference: `autoFRK` R package by Wen-Ting Wang from https://github.com/egpivo/autoFRK
"""

# development only
#import os
#import sys
#sys.path.append(os.path.abspath("./src"))

# import modules
import os
import platform
import torch
import pandas as pd
import gc
from typing import Dict, Union, Any, List
from autoFRK.utils.logger import LOGGER

# convert input into torch.Tensor recursively, using in autoFRK-Python Project
# check = ok
def to_tensor(
    obj: Any,
    dtype = torch.float64,
    device: Union[torch.device, str] = 'cpu'
) -> Union[torch.Tensor, Dict[str, Any], List[Any], None]:
    """  
    Recursively convert various Python, NumPy, or pandas objects into torch.Tensor or nested tensor structures.
    Handles: bool, numbers, list/tuple, np.ndarray, dict, torch.Tensor, pandas.Series, pandas.DataFrame, or None.

    Parameters
    ----------
    obj : Any
        Input object to be converted. Supported types include bool, int, float, list, tuple, np.ndarray, 
        dict (possibly nested), torch.Tensor, pandas.Series, pandas.DataFrame, or None.
    dtype : torch.dtype, optional
        Desired torch dtype for numeric conversion (default: torch.float64).
    device : torch.device or str, optional
        Target device for tensor allocation (default: 'cpu').

    Returns
    -------
    Union[torch.Tensor, Dict[str, Any], List[Any], None]
        A torch.Tensor, a nested dictionary/list of tensors, or None.
    """
    if isinstance(obj, torch.Tensor):
        if obj.dtype != dtype or obj.device != device:
            t = obj.to(dtype=dtype, device=device)
        else:
            t = obj
    elif isinstance(obj, bool):
        t = torch.tensor(obj, dtype=torch.bool, device=device)
    elif isinstance(obj, (int, float)):
        t = torch.tensor(obj, dtype=dtype, device=device)
    elif isinstance(obj, (list, tuple)):
        converted = [to_tensor(x, dtype=dtype, device=device) for x in obj]
        try:
            t = torch.stack(converted)
        except:
            t = converted
    elif isinstance(obj, dict):
        t = {k: to_tensor(v, dtype=dtype, device=device) for k, v in obj.items()}
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        t = torch.tensor(obj.values, dtype=dtype, device=device)
    elif hasattr(obj, 'shape'):
        t = torch.tensor(obj, dtype=dtype, device=device)
    elif obj is None:
        t = obj
    elif isinstance(obj, (torch.dtype, type, torch.device, str)):
        t = obj
    else:
        error_msg = f"Unsupported type: {type(obj)}"
        LOGGER.error(error_msg)
        raise TypeError(error_msg)
    
    return t

def clear():
    """
    Clear the terminal screen.
    """
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
cls = clear

def clear_all():
    """  
    Remove common global variables from memory.  
    Deletes variables of types: int, float, str, list, dict, bool, torch.Tensor.  
    """ 
    for name, val in list(globals().items()):
        if isinstance(val, (int, float, str, list, dict, bool, torch.Tensor)):
            del globals()[name]

def cleanup_memory():
    """  
    Force garbage collection and clear PyTorch CUDA memory cache.  
    """ 
    gc.collect()
    torch.cuda.empty_cache()
cm = cleanup_memory

def p(
    obj: Any
) -> None:
    """
    Recursively pretty-print dicts, lists/tuples, or torch.Tensors.
    Tensors are displayed with line breaks and indentation for readability.

    Parameters
    ----------
    obj : Any
        The object to print. Can be a dict, list, tuple, torch.Tensor, or any nested combination thereof.

    Returns
    -------
    None
        This function prints the formatted output directly to stdout and does not return a value.
    """
    def pretty_tensor(tensor: torch.Tensor, indent: int = 6) -> str:
        lines = str(tensor).split("\n")
        if len(lines) == 1:
            return f"{lines[0]}"
        ind = " " * indent
        return "\n" + "\n".join(ind + line for line in lines) + "\n" + " " * (indent - 2)

    def _p(obj, indent=0):
        space = " " * indent
        if isinstance(obj, dict):
            print(space + "{")
            for k, v in obj.items():
                print(f"{space}  {repr(k)}: ", end="")
                _p(v, indent + 4)
            print(space + "}")
        elif isinstance(obj, list) or isinstance(obj, tuple):
            print(space + "[")
            for v in obj:
                _p(v, indent + 4)
            print(space + "]")
        elif isinstance(obj, torch.Tensor):
            print(pretty_tensor(obj, indent + 4))
        else:
            print(repr(obj))

    _p(obj)



import os
import platform
import torch
import gc
from typing import Optional, Union, Any, Dict

dtype = torch.float64
torch.set_default_dtype(dtype) 

def clear():
    """清空終端畫面"""
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
cls = clear

def clear_all():
    """
    清除全域變數中常用的資料型態變數 (int, float, str, list, dict, torch.Tensor)
    """
    for name, val in list(globals().items()):
        if isinstance(val, (int, float, str, list, dict, torch.Tensor)):
            del globals()[name]

def cleanup_memory():
    """
    強制垃圾回收並清空 PyTorch CUDA 記憶體
    """
    gc.collect()
    torch.cuda.empty_cache()
cm = cleanup_memory

def p(obj):
    """
    遞迴漂亮印出 dict / list / tensor。
    tensor 會換行縮排對齊，方便查看。
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

import torch
import gc

dtype = torch.float64
torch.set_default_dtype(dtype) 

def convert_all_tensors_dtype(dtype=dtype):
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            if obj.dtype != dtype:
                obj.data = obj.data.to(dtype=dtype)

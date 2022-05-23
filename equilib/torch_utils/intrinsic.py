#!/usr/bin/env python3

import torch
from typing import Union

pi = torch.Tensor([3.14159265358979323846])

DataLike = Union[float, torch.Tensor]

def deg2rad(tensor: torch.Tensor) -> torch.Tensor:
    """Function that converts angles from degrees to radians"""
    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.0


def create_intrinsic_matrix(
    height: int,
    width: int,
    fov_x: DataLike,
    skew: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create intrinsic matrix

    params:
    - height, width (int)
    - fov_x (float): make sure it's in degrees
    - skew (float): 0.0
    - dtype (torch.dtype): torch.float32
    - device (torch.device): torch.device("cpu")

    returns:
    - K (torch.tensor): 3x3 intrinsic matrix
    """
    vals = [skew, width, height]
    for i in range(len(vals)):
        vals[i] = torch.tensor(vals[i], dtype=dtype, device=device).reshape(1)
    
    skew, width, height = vals
    
    fov_x = fov_x.to(dtype) if torch.is_tensor(fov_x) else torch.tensor(fov_x,dtype=dtype).reshape(1)
    f = width / (2 * torch.tan(deg2rad(fov_x) / 2))
    
    zero = torch.zeros(1)
    one = torch.ones(1)
    
    K = torch.stack([
            torch.stack([f, skew, width / 2]),
            torch.stack([zero, f, height / 2]),
            torch.stack([zero, zero, one])]).reshape(3,3)
    K = K.type(dtype)
    return K.to(device)

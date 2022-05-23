#!/usr/bin/env python3

from typing import Dict, List, Union

import numpy as np

import torch

DataLike = Union[float, torch.Tensor]

def create_global2camera_rotation_matrix(
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Rotation from global (world) to camera coordinates"""
    R_XY = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # X <-> Y
        dtype=dtype,
    )
    R_YZ = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],  # Y <-> Z
        dtype=dtype,
    )
    R = R_XY @ R_YZ
    return R.to(device)


def create_rotation_matrix(
    roll: DataLike,
    pitch: DataLike,
    yaw: DataLike,
    z_down: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create Rotation Matrix

    params:
    - roll, pitch, yaw (float): in radians
    - z_down (bool): flips pitch and yaw directions
    - dtype (torch.dtype): data types

    returns:
    - R (torch.Tensor): 3x3 rotation matrix
    """

    vals = [roll, pitch, yaw]
    for i in range(len(vals)):
        if not torch.is_tensor(vals[i]): vals[i] = torch.tensor(vals[i]).reshape(1)
    
    roll, pitch, yaw = vals
    zero = torch.zeros(1)
    one = torch.ones(1)

    # calculate rotation about the x-axis
    R_x = torch.stack([
                torch.stack([one, zero, zero]),
                torch.stack([zero, torch.cos(roll), -torch.sin(roll)]),
                torch.stack([zero, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)
    R_x = R_x.type(dtype)
    
    # calculate rotation about the y-axis
    if not z_down:
        pitch = -pitch
    R_y = torch.stack([
                torch.stack([torch.cos(pitch), zero, torch.sin(pitch)]),
                torch.stack([zero, one, zero]),
                torch.stack([-torch.sin(pitch), zero, torch.cos(pitch)])]).reshape(3,3)
    R_y = R_y.type(dtype)
    
    # calculate rotation about the z-axis
    if not z_down:
        yaw = -yaw
    R_z = torch.stack([
                torch.stack([torch.cos(yaw), -torch.sin(yaw), zero]),
                torch.stack([torch.sin(yaw), torch.cos(yaw), zero]),
                torch.stack([zero, zero, one])]).reshape(3,3)
    R_z = R_z.type(dtype)
    
    R = R_z @ R_y @ R_x
    return R.to(device)


def create_rotation_matrix_at_once(
    roll: DataLike,
    pitch: DataLike,
    yaw: DataLike,
    z_down: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create rotation matrix at once"

    params:
    - roll, pitch, yaw (float): in radians
    - z_down (bool): flips pitch and yaw directions
    - dtype (torch.dtype): data types
    - device (torch.device): torch.device("cpu")

    returns:
    - R (torch.Tensor): 3x3 rotation matrix

    NOTE: same results as `create_rotation_matrix` but a little bit faster
    """

    if not z_down:
        pitch = -pitch
        yaw = -yaw

    vals = [roll, pitch, yaw]
    for i in range(len(vals)):
        if not torch.is_tensor(vals[i]): vals[i] = torch.tensor(vals[i])
    
    roll, pitch, yaw = vals
    
    return torch.tensor(
        [
            [
                torch.cos(yaw) * torch.cos(pitch),
                torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll)
                - torch.sin(yaw) * torch.cos(roll),
                torch.cos(yaw) * torch.sin(pitch) * torch.cos(roll)
                + torch.sin(yaw) * torch.sin(roll),
            ],
            [
                torch.sin(yaw) * torch.cos(pitch),
                torch.sin(yaw) * torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll)
                + torch.cos(yaw) * torch.cos(roll),
                torch.sin(yaw) * torch.sin(pitch) * torch.cos(roll)
                - torch.cos(yaw) * torch.sin(roll),
            ],
            [
                -torch.sin(pitch),
                torch.cos(pitch) * torch.sin(roll),
                torch.cos(pitch) * torch.cos(roll),
            ],
        ],
        dtype=dtype,
        device=device,
    )


def create_rotation_matrices(
    rots: List[Dict[str, float]],
    z_down: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create rotation matrices from batch of rotations

    This methods creates a bx3x3 np.ndarray where `b` referes to the number
    of rotations (rots) given in the input
    """

    R = torch.empty((len(rots), 3, 3), dtype=dtype, device=device)
    for i, rot in enumerate(rots):
        # FIXME: maybe default to `create_rotation_matrix_at_once`?
        # NOTE: at_once is faster with cpu, while slower on GPU
        R[i, ...] = create_rotation_matrix(
            **rot, z_down=z_down, dtype=dtype, device=device
        )

    return R


def create_rotation_matrix_dep(
    x: float,
    y: float,
    z: float,
    z_down: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create Rotation Matrix

    NOTE: DEPRECATED
    """
    # calculate rotation about the x-axis
    R_x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(x), -np.sin(x)],
            [0.0, np.sin(x), np.cos(x)],
        ],
        dtype=dtype,
    )
    # calculate rotation about the y-axis
    if not z_down:
        y = -y
    R_y = torch.tensor(
        [
            [np.cos(y), 0.0, -np.sin(y)],
            [0.0, 1.0, 0.0],
            [np.sin(y), 0.0, np.cos(y)],
        ],
        dtype=dtype,
    )
    # calculate rotation about the z-axis
    if not z_down:
        z = -z
    R_z = torch.tensor(
        [
            [np.cos(z), np.sin(z), 0.0],
            [-np.sin(z), np.cos(z), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )
    R = R_z @ R_y @ R_x
    return R.to(device)

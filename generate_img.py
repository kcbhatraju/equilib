import torch
from torch import nn, optim
from torchvision import transforms, utils

import numpy as np
import json
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

from equilib.equi2pers.base import Equi2Pers as Cut

img_sets = 32
batch_size = 32
base_img = transforms.ToTensor()(Image.open("equi.png"))
img_size = (128,128)
tau = torch.tensor(6.28318530717958647692)
real_distr = [ torch.normal(0.,0.,(batch_size,)),
               torch.normal(0.,0.,(batch_size,)),
               torch.normal(100.,0.,(batch_size,)),
               torch.normal(0.,0.,(batch_size,)) ]
for i in range(img_sets):
    concat = None
    for j in range(batch_size):
        rots = {
            "roll": real_distr[0][j],
            "pitch": real_distr[1][j],
            "yaw": real_distr[3][j], # torch.rand([])*tau
        }
        
        cut = Cut(height=img_size[0],width=img_size[1],fov_x=real_distr[2][j],mode="bilinear")
        img = transforms.ToPILImage()(transforms.Grayscale()(cut(equi=base_img,rots=rots)))
        img.save(f"imgs/{i}_{j}.png")

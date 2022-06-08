import numpy as np
from PIL import Image

import torch# , torchviz
torch.set_grad_enabled(True)

from torch.distributions import Uniform, Normal
from torchvision.transforms import transforms
from equilib.equi2pers.base import Equi2Pers

img = Image.open("equi2.jpeg")
img.show()

img = transforms.ToTensor()(img)

fov = Uniform(40.,90.)
roll = Normal(0.,0.025)
pitch = Normal(-0.05,0.05)
yaw = Uniform(0.,360.)

def sample(distr):
    return distr.rsample().reshape(1).requires_grad_()

grad = False
if grad:
    params = [sample(fov), sample(roll), sample(pitch), sample(yaw)] # gradient test
else:
    params = [100, np.pi/8, np.pi/16, np.pi/16] # original library test

equi2pers = Equi2Pers(height=375,
                      width=600, 
                      fov_x=params[0],
                      mode="bilinear"
                      )
rots = {'roll': params[1],
        'pitch': params[2],
        'yaw': params[3]
        }
new_img = equi2pers(equi=img, rots=rots)

if grad:
    out = new_img.mean()
    out.backward(retain_graph=True)
    for val in params:
        print(val.grad)

    # torchviz.make_dot(new_img).render("computation_graph", format="png")

new_img = transforms.ToPILImage()(new_img)
new_img.show()
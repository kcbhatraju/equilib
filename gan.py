from os import listdir, path
import numpy as np

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch import autograd, nn, optim, no_grad, cat, empty, full, isnan, normal, randn, zeros
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from equilib.equi2pers.base import Equi2Pers # Uses equilib "equi2pers" function

autograd.set_detect_anomaly(True) # shows training nan gradient cause

# hyperparameters
batch_size = 5
num_epochs = 20
lrd = 3e-4

print("Starting image generation...")
gen_imgs = []
for i in range(5*10): # 10 epochs
    out = None
    for j in range(batch_size):
        # real image distribution; trying to make generator match this over time
        fov=normal(90.,0.02,(1,)).requires_grad_() # zoom
        roll = normal(0.,0.02,(1,)).requires_grad_() # z rotation
        pitch = normal(0.,0.02,(1,)).requires_grad_() # y rotation
        
        equi2pers = Equi2Pers(height=256,
                        width=256, 
                        fov_x=fov,
                        mode="bilinear"
                    )
        rots = {'roll': roll,
                'pitch': pitch,
                'yaw': randn(1),
                }
            
        new_img = equi2pers(equi=transforms.ToTensor()(Image.open("equi.png")), rots=rots)
        if out is None:
            out = new_img.unsqueeze(0)
        else:
            out = cat((out,new_img.unsqueeze(0)))
    gen_imgs.append(out) # 50 batches of images of size (5, 3, 256, 256)
print("Ending image generation...")

# visualize real images
real_batch = gen_imgs[0]
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch,padding=True,normalize=True),(1,2,0)))
plt.show()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.img = transforms.ToTensor()(Image.open("equi.png"))
        # initial fake image distribution
        self.fov=normal(0.,0.02,(1,)).requires_grad_() # zoom
        self.roll = normal(90,0.02,(1,)).requires_grad_() # z rotation
        self.pitch = normal(90,0.02,(1,)).requires_grad_() # y rotation
    
    def forward(self, yaw):
        output = None
        for i in range(batch_size):
            equi2pers = Equi2Pers(
                        height=256,
                        width=256, 
                        fov_x=self.fov,
                        mode="bilinear"
                    )
            rots = {'roll': self.roll,
                    'pitch': self.pitch,
                    'yaw': yaw[i],
                    }
            
            new_img = equi2pers(equi=self.img, rots=rots)
            if output is None:
                output = new_img.unsqueeze(0)
            else:
                output = cat((output,new_img.unsqueeze(0)))
        output = nn.Tanh()(output) # between -1 and 1
        return output # shape: (5, 3, 256, 256)
        


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3,4,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(4,5,4,2,1,bias=False),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(5,1,4,1,0,bias=False),
            nn.Flatten(),
            nn.Linear(3721,1),
            nn.Sigmoid() # between 0 and 1
        )
    
    def forward(self, img):
        return self.main(img)


netG = Generator()
netD = Discriminator()
# set disriminator weights randomly (mean 0, std 0.02)
classname = netD.__class__.__name__
if classname.find("Conv") != -1:
    nn.init.normal_(netD.weight.data,0.0,0.02)
elif classname.find("BatchNorm") != -1:
    nn.init.normal_(netD.weight.data,1.0,0.02)
    nn.init.constant_(netD.bias.data,0)

criterion = nn.BCELoss()
fixed_noise = empty((batch_size, 1)).normal_(mean=180,std=60)

optimD = optim.Adam(netD.parameters(), lr=lrd)

print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for i, imgs in enumerate(gen_imgs):
        print(f"Img {i+1}/{len(gen_imgs)}")
        b_size = imgs.shape[0]
        
        gen_takes = 1000 if i == 0 else 1 # give generator head start in training
        for j in range(gen_takes):
            print(f"Generator {j+1}/{gen_takes}")
            ###################################
            # (1) trainG: maximize log(D(G(z)))
            ###################################
            
            # zero gradients
            netG.fov.grad = zeros(1)
            netG.roll.grad = zeros(1)
            netG.pitch.grad = zeros(1)
            
            optimD.zero_grad()
            
            noiseG = empty((batch_size, 1)).normal_(mean=180,std=60)
            fakeNG = netG(noiseG) # forward pass causes nan values after some training
            # RuntimeError: Function 'AsinBackward0' returned nan values in its 0th output.
            # Line of code: phi = torch.asin(M[..., 2] / torch.norm(M, dim=-1))
            # equilib/equi2pers/torch.py, line 83
            print("fov:", netG.fov)
            print("roll:", netG.roll)
            print("pitch:", netG.pitch)
            
            output = netD(fakeNG).view(-1)
            print("DiscPred:", output)
            label = full((b_size,), 1.)
            errG = criterion(output, label)
            print(errG)
            errG.backward(retain_graph=True)
            
            print("fov grad:", netG.fov.grad)
            print("roll grad:", netG.roll.grad)
            print("pitch grad:", netG.pitch.grad)
            
            with no_grad():
                netG.fov -= 0.1*netG.fov.grad
                netG.roll -= 0.1*netG.roll.grad
                netG.pitch -= 0.1*netG.pitch.grad
        
        #################################################
        # (1) trainD: maximize log(D(x)) + log(1-D(G(z)))
        #################################################
        netG.fov.grad = zeros(1)
        netG.roll.grad = zeros(1)
        netG.pitch.grad = zeros(1)
            
        optimD.zero_grad()
        realD = netD(imgs).view(-1)
        print("Real:", realD)
        realL = full((b_size,), 1.)
        errD_real = criterion(realD, realL)
        print(errD_real)
        # errD_real.backward(retain_graph=True)
        
        noiseD = empty((batch_size, 1)).normal_(mean=180,std=60)
        fakeND = netG(noiseD)
        fakeD = netD(fakeND).view(-1)
        print("Fake:", fakeD)
        fakeL = full((b_size,), 0.)
        errD_fake = criterion(fakeD, fakeL)
        print(errD_fake)
        
        errD = errD_real + errD_fake
        errD.backward(retain_graph=True)
        optimD.step()

print(netG.fov)
print(netG.roll)
print(netG.pitch)
# visualize outputs after training
with no_grad():
    fake = netG(fixed_noise)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(vutils.make_grid(fake,padding=True,normalize=True),(1,2,0)))
    plt.show()

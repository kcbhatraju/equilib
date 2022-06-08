import torch
from torch import nn, optim
from torchvision import transforms
import torchvision.utils as vutils

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from equilib.equi2pers.base import Equi2Pers

def visualize(batch, type="real"):
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title(f"{type.capitalize()} Images")
    plt.imshow(np.transpose(vutils.make_grid(batch,padding=True,normalize=True),(1,2,0)))
    plt.show()

image = transforms.ToTensor()(Image.open("equi.png"))
batch_size = 5
fixed_noise = torch.randn(batch_size,1,1,1)

real_imgs = []
for _ in range(10):
    out = None
    for _ in range(batch_size):
        fov = torch.normal(90.,0.02,(1,))
        roll = torch.normal(0.,0.02,(1,))
        pitch = torch.normal(0.,0.02,(1,))
        
        equi2pers = Equi2Pers(height=256,
                        width=256, 
                        fov_x=fov,
                        mode="bilinear"
                    )
        rots = {'roll': roll,
                'pitch': pitch,
                'yaw': torch.empty(1).normal_(mean=180,std=60),
                }
            
        new_img = equi2pers(equi=image, rots=rots)
        if out is None:
            out = new_img.unsqueeze(0)
        else:
            out = torch.cat((out,new_img.unsqueeze(0)))
    
    real_imgs.append(out)

visualize(real_imgs[0])

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(1,256*32,4,1,0,bias=False),
            nn.BatchNorm2d(256*32),
            nn.ReLU(True),
            nn.ConvTranspose2d(256*32,256*16,4,2,1,bias=False),
            nn.BatchNorm2d(256*16),
            nn.ReLU(True),
            nn.ConvTranspose2d(256*16,256*8,4,2,1,bias=False),
            nn.BatchNorm2d(256*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(256*8,256*4,4,2,1,bias=False),
            nn.BatchNorm2d(256*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(256*4,256*2,4,2,1,bias=False),
            nn.BatchNorm2d(256*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(256*2,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,3,4,2,1,bias=False),
            nn.Tanh(),
        )
    
    def forward(self, noise):
        return self.main(noise)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=0.2),
            nn.Conv2d(256,256*2,4,2,1,bias=False),
            nn.BatchNorm2d(256*2),
            nn.LeakyReLU(0.2,inplace=0.2),
            nn.Conv2d(256*2,256*4,4,2,1,bias=False),
            nn.BatchNorm2d(256*4),
            nn.LeakyReLU(0.2,inplace=0.2),
            nn.Conv2d(256*4,256*8,4,2,1,bias=False),
            nn.BatchNorm2d(256*8),
            nn.LeakyReLU(0.2,inplace=0.2),
            nn.Conv2d(256*8,256*16,4,2,1,bias=False),
            nn.BatchNorm2d(256*16),
            nn.LeakyReLU(0.2,inplace=0.2),
            nn.Conv2d(256*16,256*32,4,2,1,bias=False),
            nn.BatchNorm2d(256*32),
            nn.LeakyReLU(0.2,inplace=0.2),
            nn.Conv2d(256*32,1,4,1,0,bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, img):
        return self.main(img)


netG = Generator()
netD = Discriminator()

optimG = optim.Adam(netG.parameters(), lr=0.002)
optimD = optim.Adam(netD.parameters(), lr=3e-4)
criterion = nn.BCELoss()

epochs = 2
for e in range(epochs):
    print(f"Epoch {e+1}/{epochs}")
    for real_batch in real_imgs:
        ################################
        # Train discriminator
        # max log(D(real)) + log(1-D(G(noise)))
        ################################
        
        netD.zero_grad()
        realD_pred = netD(real_batch).view(-1)
        realD_labels = torch.ones_like(realD_pred)
        errD_real = criterion(realD_pred,realD_labels)
        errD_real.backward()
        
        noise = torch.randn(batch_size,1,1,1)
        fake_batch = netG(noise)
        fakeD_pred = netD(fake_batch.detach()).view(-1)
        fakeD_labels = torch.zeros_like(fakeD_pred)
        errD_fake = criterion(fakeD_pred,fakeD_labels)
        errD_fake.backward()
        
        errD = errD_real + errD_fake
        print(f"Discriminator Loss: {errD}")
        
        optimD.step()
    
        ################################
        # Train generator
        # max log(D(G(noise)))
        ################################
        
        netG.zero_grad()
        
        fakeD_pred = netD(fake_batch).view(-1)
        fakeG_labels = torch.ones_like(fakeD_pred)
        errG = criterion(fakeD_pred,fakeG_labels)
        errG.backward()
        
        print(f"Generator Loss: {errG}")
        
        optimG.step()

fixedG_imgs = netG(fixed_noise)
visualize(fixedG_imgs, type="fake")

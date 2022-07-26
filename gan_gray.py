import torch
from torch import nn, optim
from torchvision import transforms
import torchvision.utils as vutils
# import torchviz

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

from equilib.equi2pers.base import Equi2Pers as Cut

PI = torch.tensor(3.14159265358979323846)
IMAGE = transforms.ToTensor()(Image.open("equi.png"))

def GRAYSCALE(img):
    transform = transforms.Compose([transforms.Normalize(torch.mean(img,[1,2]),torch.std(img,[1,2])),
                                    transforms.Grayscale()
                                    ])
    return transform(img)

def SHOW(batch, title):
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(batch,padding=True,normalize=True),(1,2,0)))
    plt.show()

batch_size = 5
num_epochs = 50
img_sets = 50
lrd = 3e-9

gen_imgs = []
for i in range(img_sets):
    out = None
    for j in range(batch_size):
        fov = torch.normal(90.,0.02,(1,)).requires_grad_()
        roll = torch.normal(0.,0.02,(1,)).requires_grad_()
        pitch = torch.normal(0.,0.02,(1,)).requires_grad_()
        yaw = torch.empty((1,)).uniform_(0,2*PI)
        
        lib = Cut(height=100, width=100, fov_x=fov, mode="bilinear")

        rots = {'roll': roll,
                'pitch': pitch,
                'yaw': yaw,
                }

        new_img = lib(equi=IMAGE, rots=rots)
        new_img = GRAYSCALE(new_img)
        # if (i+1==img_sets) and (j+1==batch_size): torchviz.make_dot(new_img).render("ztest_img1",format="png")
        
        out = torch.cat((out,new_img.unsqueeze(0))) if torch.is_tensor(out) else new_img.unsqueeze(0)
    
    gen_imgs.append(out)

SHOW(gen_imgs[0], "Real Images")




class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fov = torch.normal(20.,0.02,(1,)).requires_grad_()
        self.roll = torch.normal(PI/2,0.02,(1,)).requires_grad_()
        self.pitch = torch.normal(PI,0.02,(1,)).requires_grad_()

    def forward(self, yaw):
        out = None
        for i in range(batch_size):
            lib = Cut(height=100, width=100, fov_x=self.fov, mode="bilinear")
            
            rots = {'roll': self.roll,
                    'pitch': self.pitch,
                    'yaw': yaw[i],
                    }
            
            new_img = lib(equi=IMAGE, rots=rots)
            new_img = GRAYSCALE(new_img)
            # if (i+1==batch_size): torchviz.make_dot(new_img).render("ztest_img2",format="png")
            
            out = torch.cat((out,new_img.unsqueeze(0))) if torch.is_tensor(out) else new_img.unsqueeze(0)
        
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(10000,1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img = img.view(batch_size, -1)
        return self.net(img)

netG = Generator()
netD = Discriminator()

classname = netD.__class__.__name__
if classname.find("Conv") != -1:
    nn.init.normal_(netD.weight.data,0.0,0.02)
elif classname.find("BatchNorm") != -1:
    nn.init.normal_(netD.weight.data,1.0,0.02)
    nn.init.constant_(netD.bias.data,0)


fixed_noise = torch.empty((batch_size, 1)).uniform_(0,2*PI)

criterion = nn.BCELoss()
optimD = optim.Adam(netD.parameters(), lr=lrd)

SHOW(netG(fixed_noise), "Original Fake Images")

for e in range(num_epochs):
    print(f"Epoch {e+1}/{num_epochs}")
    for i in range(img_sets):
        print(f"Img {i+1}/{len(gen_imgs)}")
        mini_batch = gen_imgs[i].shape[0]
        
        gen_takes = 50 if (e == i == 0) else 5
        for j in range(gen_takes):
            netG.fov.grad = torch.zeros(1)
            netG.roll.grad = torch.zeros(1)
            netG.pitch.grad = torch.zeros(1)
            
            optimD.zero_grad()
            
            noiseG = torch.empty((mini_batch, 1)).uniform_(0,2*PI)
            fakeNG = netG(noiseG)
            
            print("FoV:", netG.fov)
            print("Roll:", netG.roll)
            print("Pitch:", netG.pitch)
            
            output = netD(fakeNG).view(-1)
            print("Discriminator Prediction (Generator):", output)
            
            label = torch.full((mini_batch,), 1.)
            errG = criterion(output, label)
            errG.backward(retain_graph=True)
            
            print("FoV Gradient:", netG.fov.grad)
            print("Roll Gradient:", netG.roll.grad)
            print("Pitch Gradient:", netG.pitch.grad)
            
            with torch.no_grad():
                netG.fov.data = torch.clamp(netG.fov.data-4.5*netG.fov.grad,min=0,max=100)
                netG.roll.data = torch.clamp(netG.roll.data-3.5*netG.roll.grad,min=0,max=2*PI)
                netG.pitch.data = torch.clamp(netG.pitch.data-2*netG.pitch.grad,min=0,max=2*PI)
        
        optimD.zero_grad()
        realD = netD(gen_imgs[i]).view(-1)
        print("Discriminator Prediction (Real):", realD)
        
        realL = torch.full((mini_batch,), 1.)
        errD_real = criterion(realD, realL)
        
        noiseD = torch.empty((mini_batch, 1)).uniform_(0,2*PI)
        fakeND = netG(noiseD)
        fakeD = netD(fakeND).view(-1)
        print("Discriminator Prediction (Fake):", fakeD)
        
        fakeL = torch.full((mini_batch,), 0.)
        errD_fake = criterion(fakeD, fakeL)
        
        errD = errD_real + errD_fake
        errD.backward(retain_graph=True)
        optimD.step()

print("Final FoV:", netG.fov)
print("Final Roll:", netG.roll)
print("Final Pitch:", netG.pitch)

SHOW(netG(fixed_noise), "Learned Fake Images")

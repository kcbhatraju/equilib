from os import listdir, path
import numpy as np

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch import autograd, nn, optim, no_grad, cat, empty, full, full_like, isnan, normal, randn, zeros
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from equilib.equi2pers.base import Equi2Pers

# autograd.set_detect_anomaly(True)


class Unsupervised(Dataset):
    def __init__(self, root):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.imgs = [transform(Image.open(path.join(root,n)).convert("RGB")) for n in listdir(root)[:32*3]]
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i]


# hyperparameters
realroot = "val_256/"
batch_size = 5
num_epochs = 20
lrg = 3e-4
lrd = 3e-4
beta = 0.5

realset = Unsupervised(realroot)
realloader = DataLoader(realset, batch_size=batch_size, shuffle=True)

# visualize training images
real_batch = next(iter(realloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Real Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch,padding=True,normalize=True),(1,2,0)))
# plt.show()

print("Starting image generation...")
gen_imgs = []
for i in range(5*10):
    out = None
    for j in range(batch_size):
        fov=normal(90.,0.02,(1,)).requires_grad_()
        roll = normal(0.,0.02,(1,)).requires_grad_()
        pitch = normal(0.,0.02,(1,)).requires_grad_()
        
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
    gen_imgs.append(out)
print("Ending image generation...")

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
        self.fov=normal(0.,0.02,(1,)).requires_grad_()
        self.roll = normal(90,0.02,(1,)).requires_grad_()
        self.pitch = normal(90,0.02,(1,)).requires_grad_()
    
    def forward(self, yaw):
        output = None
        for i in range(batch_size):
            equi2pers = Equi2Pers(height=256,
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
        output = nn.Tanh()(output)
        return output
        


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
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.main(img)


netG = Generator()
netD = Discriminator()
classname = netD.__class__.__name__
if classname.find("Conv") != -1:
    nn.init.normal_(netD.weight.data,0.0,0.02)
elif classname.find("BatchNorm") != -1:
    nn.init.normal_(netD.weight.data,1.0,0.02)
    nn.init.constant_(netD.bias.data,0)

criterion = nn.BCELoss()
fixed_noise = empty((batch_size, 1)).normal_(mean=180,std=60)

optimD = optim.Adam(netD.parameters(), lr=lrd, betas=(beta,0.999))

img_list = []
lossG = []
lossD = []
show = -1
org = [netG.fov.detach().clone(), netG.roll.detach().clone(), netG.pitch.detach().clone()]
print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for i, imgs in enumerate(gen_imgs):
        print(f"Img {i+1}/{len(gen_imgs)}")
        b_size = imgs.shape[0]
        
        gen_takes = 1000 if i == 0 else 1
        for j in range(gen_takes):
            print(f"Generator {j+1}/{gen_takes}")
            ###################################
            # (1) trainG: maximize log(D(G(z)))
            ###################################
            # print(imgs.shape)
            
            netG.fov.grad = zeros(1)
            netG.roll.grad = zeros(1)
            netG.pitch.grad = zeros(1)
            
            optimD.zero_grad()
            
            noiseG = empty((batch_size, 1)).normal_(mean=180,std=60)
            fakeNG = netG(noiseG)
            print("fov:", netG.fov)
            print("roll:", netG.roll)
            print("pitch:", netG.pitch)
            
            if(epoch==0 and i==0 and j==1): show = netG.fov.grad
            
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
                if not isnan(netG.fov.grad):
                    netG.fov -= 0.1*netG.fov.grad
                else:
                    netG.fov -= empty(netG.fov.shape).normal_(mean=0,std=0.02)
                if not isnan(netG.roll.grad):
                    netG.roll -= 0.1*netG.roll.grad
                else:
                    netG.roll -= empty(netG.roll.shape).normal_(mean=0,std=0.02)
                if not isnan(netG.pitch.grad):
                    netG.pitch -= 0.1*netG.pitch.grad
                else:
                    netG.pitch -= empty(netG.pitch.shape).normal_(mean=0,std=0.02)
        
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
        # errD_fake.backward(retain_graph=True)
        errD = errD_real + errD_fake
        errD.backward(retain_graph=True)
        optimD.step()

print(org)
print(netG.fov)
print(netG.roll)
print(netG.pitch)
print("show:", show)
# visualize outputs after training
with no_grad():
    fake = netG(fixed_noise)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(vutils.make_grid(fake,padding=True,normalize=True),(1,2,0)))
    plt.show()

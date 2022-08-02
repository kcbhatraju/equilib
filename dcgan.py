import torch
from torch import nn, optim
from torchvision import transforms, utils

import numpy as np
import json
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

from equilib.equi2pers.base import Equi2Pers as Cut


class Process:
    @classmethod
    def plot(cls,batch,title):
        with torch.no_grad():
            plt.figure(figsize=(8,8))
            plt.title(title)
            plt.imshow(np.transpose(utils.make_grid(batch,padding=True,normalize=True),(1,2,0)))
            plt.show()
    
    @classmethod
    def progress(cls,current,total,**kwargs):
        done_token, current_token = ("=", ">")
        token_arr = []
        token_arr.extend([done_token]*current)
        if (total-current): token_arr.extend([current_token])
        attrs = json.dumps(kwargs).replace('"',"")[1:-1]
        final = f"{current}/{total} [{''.join(token_arr)}{' '*max(0,total-current-1)}] - {attrs}"
        print(final,end=("\r","\n\n")[current==total])


num_epochs = 15
img_sets = 25
img_size = (100,100)
batch_size = 16
gen_lr = 0.01
discrim_lr = 0.01
base_img = transforms.ToTensor()(Image.open("equi.png"))
tau = torch.tensor(6.28318530717958647692)
noise_dim = 100
fixed_noise = torch.rand((batch_size,noise_dim))*tau
real_distr = [ torch.normal(0.,tau/16,(batch_size,)),
               torch.normal(0.,tau/16,(batch_size,)),
               torch.normal(100.,tau/16,(batch_size,)) ]
fake_distr = [ torch.normal(0.2,tau/16,(batch_size,)).requires_grad_(),
               torch.normal(0.2,tau/16,(batch_size,)).requires_grad_(),
               torch.normal(95.,tau/16,(batch_size,)).requires_grad_() ]

real_imgs = []
for _ in range(img_sets):
    concat = None
    for i in range(batch_size):
        rots = {
            "roll": real_distr[0][i],
            "pitch": real_distr[1][i],
            "yaw": torch.rand([])*tau
        }
        
        cut = Cut(height=img_size[0],width=img_size[1],fov_x=real_distr[2][i],mode="bilinear")
        img = cut(equi=base_img,rots=rots)
        if not torch.is_tensor(concat): concat = img.unsqueeze(0)
        else: concat = torch.cat((concat,img.unsqueeze(0)))
    
    real_imgs.append(concat) # concat.shape = [16, 3, 100, 100]


class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(noise_dim,int(np.prod(np.array(img_size)/4)*256))
        self.bn0 = nn.BatchNorm2d(256)
        self.upconv1 = nn.ConvTranspose2d(256,128,(5,5),stride=1,padding=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.upconv2 = nn.ConvTranspose2d(128,64,(5,5),stride=2,padding=2,output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.upconv3 = nn.ConvTranspose2d(64,3,(5,5),stride=2,padding=2,output_padding=1)
    
    def forward(self,yaw):
        # input shape = [16, 100]
        yaw = self.dropout(self.dense(yaw)) # yaw.shape = [16, 25*25*256]
        yaw = self.leakyrelu(self.bn0(yaw.reshape(batch_size,-1,*np.int_(np.array(img_size)/4)))) # yaw.shape = [16, 256, 25, 25]
        
        yaw = self.leakyrelu(self.bn1(self.dropout(self.upconv1(yaw)))) # yaw.shape = [16, 128, 25, 25]
        yaw = self.leakyrelu(self.bn2(self.dropout(self.upconv2(yaw)))) # yaw.shape = [16, 64, 50, 50]
        yaw = self.upconv3(yaw) # yaw.shape = [16, 3, 100, 100]
        return yaw


class Discrim(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3,64,(5,5),stride=1,padding=2)
        self.conv2 = nn.Conv2d(64,128,(5,5),stride=2,padding=2)
        self.conv3 = nn.Conv2d(128,256,(5,5),stride=2,padding=2)
        self.dense = nn.Linear(int(np.prod(np.array(img_size)/4)*256),1)
    
    def forward(self,img):
        # input shape = [16, 3, 100, 100]
        img = self.leakyrelu(self.conv1(img)) # img.shape = [16, 64, 100, 100]
        img = self.leakyrelu(self.conv2(img)) # img.shape = [16, 128, 50, 50]
        img = self.leakyrelu(self.conv3(img)) # img.shape = [16, 256, 25, 25]
        
        img = self.flatten(img) # img.shape = [16, 256*25*25]
        img = self.sigmoid(self.dense(img)) # img.shape = [16, 1]
        
        return img


gen, discrim = Gen(), Discrim()
loss = nn.BCELoss()
gen_optim = optim.Adam(gen.parameters(),lr=gen_lr)
discrim_optim = optim.Adam(discrim.parameters(),lr=discrim_lr)

# Process.plot(real_imgs[0],"Real Images")
# Process.plot(gen(fixed_noise),"Original Fake Images")

# IDENTITY LOSS
# IMAGE-TO-IMAGE GAN / AUTOENCODERS

# autograd.set_detect_anomaly(True)
for e in range(num_epochs):
    print(f"Epoch {e+1}/{num_epochs}")
    for i in range(img_sets):
        # train generator
        gen_optim.zero_grad()
        fake_noise = torch.rand((batch_size,noise_dim))*tau
        fake_batch = gen(fake_noise)
        gen_fake_pred = discrim(fake_batch)
        gen_labs = torch.ones_like(gen_fake_pred)
        gen_loss = loss(gen_fake_pred,gen_labs)
        gen_loss.backward(retain_graph=True)
        gen_optim.step()
        
        # train discriminator
        discrim_optim.zero_grad()
        real_batch = real_imgs[i]
        real_pred = discrim(real_batch)
        real_labs = torch.ones_like(real_pred)
        discrim_real_loss = loss(real_pred,real_labs)
        
        fake_batch = gen(fake_noise)
        fake_pred = discrim(fake_batch)
        fake_labs = torch.zeros_like(fake_pred)
        discrim_fake_loss = loss(fake_pred,fake_labs)
        
        discrim_loss = discrim_real_loss + discrim_fake_loss
        discrim_loss.backward(retain_graph=True)
        discrim_optim.step()
        
        with torch.no_grad():
            Process.progress(i+1,img_sets,
                             discrim_loss=round(discrim_loss.mean().item(),4),
                             gen_loss=round(gen_loss.mean().item(),4))
        
        """roll=round(gen.distr[0].mean().item(),4),
        pitch=round(gen.distr[0].mean().item(),4),
        fov=round(gen.distr[2].mean().item(),4))"""

Process.plot(gen(fixed_noise),"Learned Fake Images")

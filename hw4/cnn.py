import h5py
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from visualize import draw_point_cloud
from compare import *

import cv2 


class mBlock(nn.Module):
    def __init__(self, c_in, c_out,s = 1):
        super(mBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, s, 1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, 1, 1),
            nn.BatchNorm2d(c_out)
        )
        
        if s != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, s, 1),
                nn.BatchNorm2d(c_out)
                )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        #print("in forward")
        #print("x shape ",x.shape)
        out = self.conv(x)
        #print("conv shape ",out.shape)
        out += self.shortcut(x)
        out = F.relu(out)
        #print("forward finish")
        return out
    

class mUpblock(nn.Module):
    def __init__(self, c_in, c_out,s = 1):
        super(mUpblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 5, s, 2),
            nn.ReLU(inplace=True),
        )
        
    def forward(self,x):
        #print("===== in up pooling, size ", x.shape)
        out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        #print("     in up pooling, size ", out.shape)
        out = self.conv(out)
        #print("----- in up pooling, size ", out.shape)
        #print(" ")
        return out
        
        
class mNet(nn.Module):
    def __init__(self, n_class=21):
        super(mNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 3,  stride = 2, padding = 1),
        )
        self.conv2 = nn.Sequential(
            mBlock(64, 64),
            mBlock(64, 64),
        )
        self.conv3 = nn.Sequential(
            mBlock(64, 128, 2),
            mBlock(128, 128),
        )
        self.conv4 = nn.Sequential(
            mBlock(128, 256, 2),
            mBlock(256, 256),
        )
        self.conv5 = nn.Sequential(
            mBlock(256, 512, 2),
            mBlock(512, 512),
        )
        self.conv6 = nn.Sequential(
            mUpblock(512,256),
            mUpblock(256,128),
            mUpblock(128,64),
            mUpblock(64,32),
            nn.Conv2d(32, 1, 3, 1, 1),
        )

        
    def forward(self,x):
        #print("conv0, ",x.shape)
        x = self.conv1(x)
        #print("conv1, ",x.shape)
        x = self.conv2(x)
        #print("conv2, ",x.shape)
        x = self.conv3(x)
        #print("conv3, ",x.shape)
        x = self.conv4(x)
        #print("conv4, ",x.shape)
        x = self.conv5(x)
        #print("conv5, ",x.shape)
        x = self.conv6(x)
        #print("conv6, ",x.shape)
        return torch.relu(x)
  
class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        eps = 1e-7
        d = torch.log(x+eps)-torch.log(y)
        return torch.mean(torch.pow(d, 2))-0.5*torch.pow(torch.mean(d),2)

def get_data_nyu(bs, shuff =  True):

    img_res = np.load('img_res.npy')
    dep_res = np.load('dep_res.npy')
    #mat = h5py.File('nyu_depth_v2_labeled.mat')
    split = sio.loadmat('splits.mat')
    
    #imgs = np.transpose(mat['images'],(0,3,2,1))
    #depths = np.transpose(mat['depths'],(0,2,1))

    f_train = np.concatenate(split['trainNdxs'])
    f_test = np.concatenate(split['testNdxs'])

    train_img = [img_res[index-1] for index in f_train]
    test_img = [img_res[index-1] for index in f_test]

    train_depth = [dep_res[index-1] for index in f_train]
    test_depth = [dep_res[index-1] for index in f_test]

    train_ds = TensorDataset(torch.tensor(train_img), torch.tensor(train_depth))
    valid_ds = TensorDataset(torch.tensor(test_img), torch.tensor(test_depth))

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle = shuff)
    test_dl = DataLoader(valid_ds, batch_size=bs, shuffle = shuff)
    return train_dl,test_dl

def train_nyu(load = False, epochs = 200, lr = 0.001):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    batch_size = 32

    if load:
        model = torch.load('model_si.pkl')
    else:
        model = mNet().to(dev)

    train_dl,test_dl = get_data_nyu(batch_size)
    optimizer = optim.SGD(model.parameters(),lr = lr,momentum = 0.9)
    criterion = My_loss()
    

    loss = 0.0
    for epoch in range(epochs):
        for i, data in enumerate(train_dl):
            (x,y) =  data
            #print(x.shape,y.shape)
            x = x.to(dev, dtype = torch.float)
            y = y.view(-1,1,y.shape[1],y.shape[2])
            y = y.to(dev, dtype = torch.float)
            #print("y, ",y.shape)
            outputs = model(x)
            

            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if i%10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, loss.item())) 
    
    torch.save(model,'model_si.pkl')
    #net = torch.load('model.pkl')
def draw_nyu():
    dev = torch.device("cpu") 
    net = torch.load('model_si.pkl',map_location='cpu')
    train_dl,test_dl = get_data_nyu(32, False)

    for i, data in enumerate(train_dl):
        (x,y) =  data
        x = x.to(dev, dtype = torch.float)
        y = y.view(-1,1,y.shape[1],y.shape[2])
        y = y.to(dev, dtype = torch.float)
        outputs = net(x)

        original = x[0].detach().numpy().transpose()
        original_re = cv2.resize(original,(640,480))
        out = outputs[0].detach().numpy().transpose()
        out_re = cv2.resize(out,(640,480))
        gt = y[0].detach().numpy().transpose()
        gt_re = cv2.resize(gt,(640,480))
        plt.figure()
        
        plt.subplot(131)
        plt.title('Original image') 
        plt.imshow(original.astype(int))

        plt.subplot(132)
        plt.title('Result from CNN') 
        plt.imshow(out)

        plt.subplot(133)
        plt.title('Ground truth') 
        plt.imshow(gt)

        plt.show()

        draw_point_cloud(original_re,out_re,1)
        draw_point_cloud(original_re,gt_re,1)
        break


def test_nyu():
    dev = torch.device("cpu") 
    net = torch.load('model_si.pkl',map_location='cpu')

    train_dl,test_dl = get_data_nyu(32)
    count = 0 
    t1 = 0
    t2 = 0
    t3 = 0
    absRe = 0
    sqrRe = 0
    rmseLi = 0
    rmseLo = 0
    rmseSi = 0
    for i, data in enumerate(train_dl):
        (x,y) =  data
        x = x.to(dev, dtype = torch.float)
        y = y.view(-1,1,y.shape[1],y.shape[2])
        y = y.to(dev, dtype = torch.float)
        outputs = net(x)

        for idx in range(outputs.shape[0]):

            pred = outputs[idx].detach().numpy()
            gt = y[idx].detach().numpy()

            count += 1
            t1 += threhold(pred,gt,1.25)
            t2 += threhold(pred,gt,1.25*1.25)
            t3 += threhold(pred,gt,1.25*1.25*1.25)
            absRe += AbsRe(pred,gt)
            sqrRe += SqrRe(pred,gt)
            rmseLi += RMSELinear(pred,gt)
            rmseLo += RMSELog(pred,gt)
            rmseSi += RMSESI(pred,gt)    
    print(t1/count,t2/count,t3/count,absRe/count,sqrRe/count,rmseLi/count,rmseLo/count,rmseSi/count)



#train_nyu(load = False,epochs = 200, lr = 0.001)
#test_nyu()
draw_nyu()
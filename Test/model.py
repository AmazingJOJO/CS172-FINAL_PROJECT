import torch
#import torchvision
#import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import lsd
import cv2
import numpy as np

   
    
def get_images(image):
    img = []
    name = lsd.FindApple(image)

    im = cv2.imread('000009.jpg')
    resized = cv2.resize(im, (224,224), interpolation = cv2.INTER_AREA)
    img.append(resized)
    
    #print(name)
    #name = ['010.png','000002.jpg','1.png','1.PNG','2.png']
    #name = ['iphone 11','iphone8plus','11pro','11pro']
    for each in name:
        
        resized = cv2.resize(each, (224,224), interpolation = cv2.INTER_AREA)
        img.append(resized)
    return img






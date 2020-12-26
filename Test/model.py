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

   
    
def get_images():
    img = []
    name = lsd.FindApple("Apple.png")

    im = cv2.imread('000004.jpg')
    resized = cv2.resize(im, (224,224), interpolation = cv2.INTER_AREA)
    img.append(resized)
    
    #print(name)
    #name = ['010.png','000002.jpg','1.png','1.PNG','2.png']
    #name = ['iphone 11','iphone8plus','11pro','11pro']
    for each in name:
        
        resized = cv2.resize(each, (224,224), interpolation = cv2.INTER_AREA)
        img.append(resized)
    return img




#training(100)

'''
model = torch.load('best_model_50.pkl',map_location='cpu')
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(dev)
train_dl,test_dl = load_data(32)
print(testing(model,test_dl))
'''

classes = np.load('classes.npy',allow_pickle=True)
print(classes)

images = []
model = torch.load('best_model_50.pkl',map_location='cpu')
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(dev)


images = get_images()
images = np.asarray(images)

#resized = np.asarray(resized)

x = torch.from_numpy(images).float() / 255
x = np.transpose(x,(0,3,2,1))
with torch.no_grad():
    x = x.to(dev, dtype = torch.float)
    outputs = model(x)
    print(outputs)
    _, predicted = torch.max(outputs.data, 1)
    print(classes[predicted[1:]])
#print(testing(model,test_dl))

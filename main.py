import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import cv2
import lsd
#training(100)

'''
model = torch.load('best_model_50.pkl',map_location='cpu')
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(dev)
train_dl,test_dl = load_data(32)
print(testing(model,test_dl))
'''
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='FullHead.mhd.')
    args = parser.parse_args()
    imagePath = args.filename
    
    classes = ['iPhone8','iPhoneXR','iPhone11/12','iPhone8Plus','iPhone11Pro/12Pro','iPhoneX/Xs']
    print(classes)

    images = get_images(imagePath)
    m_images = np.asarray(images)
    x = torch.from_numpy(m_images).float() / 255
    x = np.transpose(x,(0,3,2,1))


    dev = torch.device("cpu")

    model1 = torch.load('cnn1.pkl',map_location='cpu')

    model1 = model1.to(dev)

    model2 = torch.load('cnn2.pkl',map_location='cpu')
    model2 = model2.to(dev)


    with torch.no_grad():
        x = x.to(dev, dtype = torch.float)
        outputs = model1(x)
        print(outputs[1:])
        _, predicted1 = torch.max(outputs.data, 1)
        print(predicted1[1:])

        out2 = model2(x)
        print(out2[1:])
        maxp, predicted2 = torch.max(out2.data, 1)
        for i in range(len(predicted1[1:])):
            if predicted1[1+i] == 0 and maxp[i+1] > 2.5:

                print 'image ',i, 'is ',classes[predicted2[1+i]]

    for i in range(1,len(images)):
        cv2.imshow("camera"+ str(i), images[i]) 
    #cv2.waitKey(0)


    #print(testing(model,test_dl))

main()
import os
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms as tfs
import numpy as np
classes = []


path1 = "Camera"
path2 = "TEST"

X = []
Y = []
label = 0

im_aug = tfs.Compose([
    tfs.ToPILImage(),
    tfs.RandomHorizontalFlip(),
    tfs.RandomVerticalFlip(),
    tfs.RandomRotation(45),
    tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
])
X.append([])
Y.append([])
for filename in os.listdir(path1):
    classes.append(filename)
    curPath = path1+'/'+filename

label += 1
X.append([])
Y.append([])
for filename in os.listdir(path2):
    classes.append(filename)
    curPath = path2+'/'+filename
    for image in os.listdir(curPath):
        img = cv2.imread(curPath+'/'+image)
        resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
        X[label].append(resized)
        Y[label].append(label)
label += 1


X_train = []
X_test = []
Y_train = []
Y_test = []
for i in range(0,label):
    x_train,x_test,y_train,y_test = train_test_split(X[i],Y[i],test_size=0.3)
    X_train.extend(x_train)
    X_test.extend(x_test)
    Y_train.extend(y_train)
    Y_test.extend(y_test)
print(len(X_train),len(Y_train),len(X_test),len(Y_test))
np.save('X_train.npy',X_train)
np.save('X_test.npy',X_test)
np.save('Y_train.npy',Y_train)
np.save('Y_test.npy',Y_test)
#np.save('classes.npy',classes)

import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
classes = []


path = "../DataSet"

X = []
Y = []
label = 0
for filename in os.listdir(path):
    classes.append(filename)
    X.append([])
    Y.append([])

    curPath = path+'/'+filename
    for image in os.listdir(curPath):
        img = cv2.imread(curPath+'/'+image)
        #print(img.shape)
        resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
        #print(resized.shape)
        X[label].append(resized)
        Y[label].append(label)
    label += 1

'''
X_train = []
X_test = []
Y_train = []
Y_test = []
for i in range(0,label):
    x_train,x_test,y_train,y_test = train_test_split(X[i],Y[i],train_size = 70)
    X_train.extend(x_train)
    X_test.extend(x_test)
    Y_train.extend(y_train)
    Y_test.extend(y_test)

np.save('X_train.npy',X_train)
np.save('X_test.npy',X_test)
np.save('Y_train.npy',Y_train)
np.save('Y_test.npy',Y_test)
np.save('classes.npy',classes)

'''

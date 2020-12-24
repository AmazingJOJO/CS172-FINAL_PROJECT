import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import numpy as np

def training(epochs = 200):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(dev)
    batch_size = 32

    model = torchvision.models.resnet18(num_classes = 6) 
    #model = torch.load('model.pkl')
    model = model.to(dev)

    train_dl,test_dl = load_data(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_accu = 0
    for epoch in range(epochs):
        for i, data in enumerate(train_dl):
            (x,y) =  data
            x = x.to(dev, dtype = torch.float)
            y = y.to(dev, dtype = torch.long)
            outputs = model(x)

            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if i%10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, loss.item()))
            accu = testing(model,test_dl)
            if accu > best_accu:
                best_accu = accu
                torch.save(model,'best_model.pkl')
                print("accuracy is ",accu)
    torch.save(model,'model.pkl')
                
                


def load_data(bs):
    x_train = np.load('X_train.npy')
    x_test = np.load('X_test.npy')
    y_train = np.load('Y_train.npy')
    y_test = np.load('Y_test.npy')


    x_train = np.transpose(x_train,(0,3,2,1))
    x_test = np.transpose(x_test,(0,3,2,1))

    tensor_x_train = torch.from_numpy(x_train).float() / 255
    tensor_x_test = torch.from_numpy(x_test).float() / 255

    train_ds = TensorDataset(tensor_x_train, torch.tensor(y_train))
    test_ds = TensorDataset(tensor_x_test, torch.tensor(y_test))

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle = True)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle = True)
    return train_dl,test_dl

def testing(model,test_dl):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    batch_size = 32
    
    #model = torchvision.models.resnet50(num_classes = 6) 
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dl:
            x,y = data
            x = x.to(dev, dtype = torch.float)
            y = y.to(dev, dtype = torch.long)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = correct / total
    return accuracy
    
    

training()


#model = torch.load('best_model.pkl')
#dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#model = model.to(dev)
#train_dl,test_dl = load_data(32)
#print(testing(model,test_dl))


import lsd
import sys
import cv2
import os
path = 'DataSet'

def findAllfile(path, allfile):
    filelist =  os.listdir(path)  
    for filename in filelist:  
        filepath = os.path.join(path, filename)  
        if os.path.isdir(filepath):
            #print(filepath)  
            
            findAllfile(filepath, allfile)  
        else:
            #print(filepath)
            
            crop_img =lsd.FindApple(filepath)
            cv2.imwrite('TEST' + filepath[7:], crop_img)
            print('TEST' + filepath[7:])
findAllfile(path,[])


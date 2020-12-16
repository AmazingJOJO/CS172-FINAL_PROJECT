import cv2
import numpy as np
import os
from pylsd.lsd import lsd

def Gaussian_Blur(gray):
    blurred = cv2.GaussianBlur(gray, (9, 9),0)
    
    return blurred

def FindApple(img):

    src = cv2.imread(img, cv2.IMREAD_COLOR)
    
    cv2.waitKey(0)
    height = src.shape[1]
    weight = src.shape[0]    
    image = cv2.imread(img, cv2.IMREAD_COLOR)
    src = Gaussian_Blur(image)
    cv2.imshow("blur",src)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blur = Gaussian_Blur(gray)
    lines = lsd(blur)
    for i in xrange(lines.shape[0]):
        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))
        width = lines[i, 4]
        cv2.line(blur, pt1, pt2, (255,255, 255), int(np.ceil(width/2)))
        #print(pt1,pt2)
    #gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    ret, binary = cv2.threshold(blur,200,255,cv2.THRESH_BINARY)  
    cv2.imshow("img2", binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print("coun:",contours[0],contours[1])

    c1 = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    c2 = sorted(contours, key=cv2.contourArea, reverse=True)[1]
    #print("c:",c[0],c[1],c[2],c[3])
    # compute the rotated bounding box of the largest contour

    x1, y1, w1, h1 = cv2.boundingRect(c1)
    x2, y2, w2, h2 = cv2.boundingRect(c2)
    if (w1 * h1) / (height * weight) > 0.95:
        #print(w1, h1)
        #print(height,weight)
        crop_img = image[y2:y2 + h2, x2:x2 + w2]
    else:
        crop_img = image[y1:y1 + h1, x1:x1 + w1]
    for c in contours:
        # find bounding box coordinates
        
        x, y, w, h = cv2.boundingRect(c)
        
        cv2.rectangle(src, (x,y), (x+w, y+h), (0, 255, 0), 2) 
    #cv2.rectangle(src, (x,y), (x+w, y+h), (0, 255, 0), 2)
    #draw_img = cv2.drawContours(src.copy(), [box], -1, (0, 0, 255), 2)
    """Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    crop_img = src[y1:y1+hight, x1:x1+width]"""
    #print(src.shape)
    cv2.imshow("contour", src)
    #cv2.waitKey(0)
    return crop_img

img ='0025.jpg'
i = FindApple(img)
cv2.imshow("result", i)
cv2.waitKey(0)
import cv2
import numpy as np
import os
from pylsd.lsd import lsd

class contourImage():
    def __init__(self, flagList,tag):
        self.flagList = flagList
        self.tag = []
        for i in range(len(self.flagList)):
            self.tag.append(tag)
            

def Gaussian_Blur(gray):
    blurred = cv2.GaussianBlur(gray, (7, 7),0)
    
    return blurred

def FindApple(img):

    src = cv2.imread(img, cv2.IMREAD_COLOR)
    
    cv2.waitKey(0)
    height = src.shape[1]
    weight = src.shape[0]    
    image = cv2.imread(img, cv2.IMREAD_COLOR)
    src = Gaussian_Blur(image)
    #cv2.imshow("blur",src)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blur = Gaussian_Blur(gray)
    lines = lsd(blur)
    for i in xrange(lines.shape[0]):
        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))
        width = lines[i, 4]
        cv2.line(blur, pt1, pt2, (0,0,0), int(np.ceil(width*1.5)))
        #print(pt1,pt2)
    #gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    ret, binary = cv2.threshold(blur, 5, 255, cv2.THRESH_BINARY)
    
    cv2.imshow("img2", binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print("coun:",contours[0],contours[1])

    """c1 = sorted(contours, key=cv2.contourArea, reverse=True)[0]
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
        crop_img = image[y1:y1 + h1, x1:x1 + w1]"""
    c_all = sorted(contours, key = cv2.contourArea, reverse = True)
    x1, y1, w1, h1 = cv2.boundingRect(c_all[0])
    
    if float((w1 * h1)) / float((height * weight)) > 0.95:  #image itself
        print("cut")
        c_all = c_all[1:]
    c_new = []
    conImgList = contourImage(c_all, False)
    
    for i in range(len(conImgList.flagList)):
        x1, y1, w1, h1 = cv2.boundingRect(conImgList.flagList[i])
        #print(w1,h1)
        for j in range(i+1,len(conImgList.flagList)):
            x2, y2, w2, h2 = cv2.boundingRect(conImgList.flagList[j])
            if ((x2 > x1 and x2 < x1 + w1) and (y2 > y1 and y2  < y1 + h1) or ((x1 > x2 and x1 < x2 + w2) and (y1 > y2 and y1  < y2 + h2))) and (w2*h2 < w1*h1):  #contour c2 is in c1
                #print("inside bigger contour")
                conImgList.tag[j] = True

    for k in range(len(conImgList.flagList)):
        if conImgList.tag[k] == False:
            c_new.append(conImgList.flagList[k])        
    
    c_new = sorted(c_new, key = cv2.contourArea, reverse = True)
    print(len(c_new))    
    count = 0
    crop_imgs = []
    while True:
        if count == 0:
            xi, yi, wi, hi = cv2.boundingRect(c_new[count])
            crop_img = image[yi:yi + hi, xi:xi + wi]
            crop_imgs.append(crop_img)


        x1, y1, w1, h1 = cv2.boundingRect(c_new[count])
        x2, y2, w2, h2 = cv2.boundingRect(c_new[count + 1])
        if float(w2 * h2) / float(w1 * h1) > 0.5:
            crop_img = image[y2:y2 + h2, x2:x2 + w2]
            crop_imgs.append(crop_img)
            count += 1
        else:
            break
    count = 0
    """for rawdata in crop_imgs:
        if crop_imgs[]"""
    result = []
    
    for crop_img_i in crop_imgs:
        crop_img_i = crop_img_i[0:crop_img_i.shape[0] / 3, 0:crop_img_i.shape[1] / 2]
        
        result.append(crop_img_i)

    for c in c_new:
        # find bounding box coordinates
        
        x, y, w, h = cv2.boundingRect(c)
        
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 4)
    
    for c in contours:
        # find bounding box coordinates
        
        x, y, w, h = cv2.boundingRect(c)
        
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #print(crop_img.shape)
    #crop_img =crop_img[0:crop_img.shape[0]/3,0:crop_img.shape[1]/2]
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
    #return result
    #print(len(result))
    """for i in range(len(result)):
        cv2.imshow("camera",result[i])"""
    return crop_imgs

img ='Apple2.png'
images = FindApple(img)
#print(len(images))
count = 0
for i in images:
    cv2.imshow("result " + str(count), i)
    count += 1
    
cv2.waitKey(0)

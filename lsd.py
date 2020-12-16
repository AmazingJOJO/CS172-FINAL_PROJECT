import cv2
import numpy as np
import os
from pylsd.lsd import lsd
fullName = '000002.jpg'
folder, imgName = os.path.split(fullName)
src = cv2.imread(fullName, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
lines = lsd(gray)
for i in xrange(lines.shape[0]):
    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
    pt2 = (int(lines[i, 2]), int(lines[i, 3]))
    width = lines[i, 4]
    cv2.line(src, pt1, pt2, (0,0, 0), int(np.ceil(width / 2)))
    #print(pt1,pt2)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.imshow("img2", gray) 
ret, binary = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)  
 
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#print("coun:",contours[0],contours[1])
c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
#print("c:",c[0],c[1],c[2],c[3])
# compute the rotated bounding box of the largest contour
x, y, w, h = cv2.boundingRect(c)
"""for c in contours:
    # find bounding box coordinates
    
    x, y, w, h = cv2.boundingRect(c)
    
    cv2.rectangle(src, (x,y), (x+w, y+h), (0, 255, 0), 2) """
cv2.rectangle(src, (x,y), (x+w, y+h), (0, 255, 0), 2)
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



crop_img = src[y:y + h, x:x + w]
cv2.imshow("or",src)
cv2.imshow("result", crop_img)
#cv2.imshow("contour", draw_img)  
cv2.waitKey(0)  
cv2.imwrite(os.path.join(folder, 'cv2_' + imgName.split('.')[0] + '.jpg'), src)
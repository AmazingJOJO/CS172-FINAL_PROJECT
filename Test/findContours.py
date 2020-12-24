import cv2
import matplotlib.pyplot as plt

image = cv2.imread('000062.jpg')
image_BGR = image.copy()

# 将图像转换成灰度图像,并执行图像高斯模糊,以及转化成二值图像
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
image_binary = cv2.Canny(blurred, 30, 100, apertureSize=3)  

#image_binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# 从二值图像中提取轮廓
# contours中包含检测到的所有轮廓,以及每个轮廓的坐标点
contours = cv2.findContours(image_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# 遍历检测到的所有轮廓,并将检测到的坐标点画在图像上
# c的类型numpy.ndarray,维度(num, 1, 2), num表示有多少个坐标点
for c in contours:
    cv2.drawContours(image, [c], -1, (255, 0, 0), 2)

image_contours = image

# display BGR image
plt.subplot(1, 3, 1)
plt.imshow(image_BGR)
plt.axis('off')
plt.title('image_BGR')

# display binary image
plt.subplot(1, 3, 2)
plt.imshow(image_binary, cmap='gray')
plt.axis('off')
plt.title('image_binary')

# display contours
plt.subplot(1, 3, 3)
plt.imshow(image_contours)
plt.axis('off')
plt.title('{} contours'.format(len(contours)))

plt.show()
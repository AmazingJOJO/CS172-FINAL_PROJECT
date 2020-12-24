import cv2
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox

img = cv2.imread('000033.jpg')

bbox, label, conf = cv.detect_common_objects(img)

output_image = draw_bbox(img, bbox, label, conf)

cv2.imshow(output_image)
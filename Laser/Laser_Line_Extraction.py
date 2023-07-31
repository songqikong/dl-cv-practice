import cv2
import numpy as np
import math

img = cv2.imread('./img/19-1.bmp')
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# filtering red area of hue
redHueArea = 15
redRange = ((hsv[:, :, 0] + 360 + redHueArea) % 360)
hsv[np.where((2 * redHueArea) > redRange)] = [0, 0, 0]
# filtering by saturation
hsv[np.where(hsv[:, :, 1] < 95)] = [0, 0, 0]
# convert to rgb
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
# select only red grayscaled channel with low threshold
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
gray = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)[1]
cv2.imshow('img',gray)
# contours processing
# (_, contours, _) = cv2.findContours(gray.copy(), cv2.RETR_LIST, 1)
# (_, contours) = cv2.findContours(gray.copy(), cv2.RETR_LIST, 1)
(contours, _) = cv2.findContours(gray.copy(), cv2.RETR_LIST, 1)
for c in contours:
    area = cv2.contourArea(c)
    if area < 8: continue
    epsilon = 0.1 * cv2.arcLength(c, True) # tricky smoothing to a single line
    approx = cv2.approxPolyDP(c, epsilon, True)
    cv2.drawContours(img, [approx], -1, [255, 255, 255], -1)

cv2.imshow('result', img)
cv2.waitKey(0)
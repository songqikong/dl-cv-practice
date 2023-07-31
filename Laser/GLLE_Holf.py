import cv2
import numpy as np

# Read the image
img = cv2.imread('./img/16.bmp')

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for green color in HSV
lower_green = np.array([40, 40, 40])  # Lower bounds for H, S, V respectively
upper_green = np.array([80, 255, 255])  # Upper bounds for H, S, V respectively

# Create a mask for green regions in the image
green_mask = cv2.inRange(hsv, lower_green, upper_green)

# Apply Canny edge detection to the green mask
edges = cv2.Canny(green_mask, 30, 90)  # Adjust the threshold values here

# Find lines using Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=5)  # Adjust the threshold value here

# Draw the lines on the original image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show the result
cv2.namedWindow('GLLE', 0)
cv2.resizeWindow("GLLE", 800, 600)
cv2.imshow('GLLE', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
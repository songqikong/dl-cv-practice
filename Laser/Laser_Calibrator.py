import cv2
import numpy as np

def calibrate_laser_plane(img_path, laser_normal_vector):
    # Read the image
    img = cv2.imread(img_path)

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

    # Calculate the centroid of the detected lines
    if lines is not None:
        x_sum = 0
        y_sum = 0
        num_lines = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_sum += x1 + x2
            y_sum += y1 + y2
        centroid_x = x_sum / (2 * num_lines)
        centroid_y = y_sum / (2 * num_lines)

        # Calculate the direction vector of the laser line
        direction_vector = np.array([centroid_x, centroid_y, 1])

        # Normalize the direction vector
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # Calculate the angle between the laser line and the laser plane
        angle_rad = np.arccos(np.dot(direction_vector, laser_normal_vector))

        # Calculate the distance between the camera and the laser plane
        distance = np.linalg.norm(direction_vector - laser_normal_vector)

        # Output the results
        print("Laser Plane Normal Vector:", laser_normal_vector)
        print("Detected Laser Line Direction Vector:", direction_vector)
        print("Angle between Laser Line and Laser Plane:", angle_rad)
        print("Distance between Camera and Laser Plane:", distance)


# Example usage:
# Assuming the laser plane's normal vector is [0, 0, 1] (perpendicular to the camera)
img_path = './img/16.bmp'
laser_normal_vector = np.array([0, 0, 1])
calibrate_laser_plane(img_path, laser_normal_vector)
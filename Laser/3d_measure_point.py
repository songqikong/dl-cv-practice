import numpy as np
import cv2

def detect_laser_point(image_path):
    # 读取图像
    img = cv2.imread(image_path)

    # 转换为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 设定阈值，将激光点二值化
    _, binary_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)

    # 使用findContours找到激光点轮廓
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 选取最大轮廓作为激光点
    if len(contours) > 0:
        laser_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(laser_contour)
        laser_x = int(M["m10"] / M["m00"])
        laser_y = int(M["m01"] / M["m00"])

        return laser_x, laser_y
    else:
        print("未检测到激光点。")
        return None

def calculate_3d_coordinates(laser_x, laser_y, camera_matrix, laser_plane_distance):
    # 将输入数据转换为numpy数组并确保数据类型为float32
    laser_x = np.array(laser_x, dtype=np.float32)
    laser_y = np.array(laser_y, dtype=np.float32)

    # 构建像素坐标数组
    pixel_coordinates = np.column_stack((laser_x, laser_y))

    # 使用undistortPoints函数将像素坐标转换为归一化坐标
    normalized_coordinates = cv2.undistortPoints(pixel_coordinates.reshape(-1, 1, 2), camera_matrix, None, None)

    # 将归一化坐标转换为相机坐标系下的三维坐标
    ones_column = np.ones((normalized_coordinates.shape[0], 1), dtype=np.float32)
    camera_coordinates = np.hstack((normalized_coordinates[:, 0, :], ones_column))
    camera_coordinates *= laser_plane_distance

    return camera_coordinates

if __name__ == "__main__":
    # 相机内参矩阵，需要根据实际相机进行设置
    camera_matrix = np.array([[496.57849394, 0, 278.74782676],
                              [0, 510.37523816, 264.13207105],
                              [0, 0, 1]], dtype=np.float32)

    # 已知激光平面与相机的距离（单位为米）
    laser_plane_distance = 1.4132776486977137
    # 假设激光与相机垂直情况下测定好的

    image_path = "./img/19-1.bmp"  # 根据实际情况设置图片路径

    # 检测激光点的像素坐标
    laser_x, laser_y = detect_laser_point(image_path)

    if laser_x is not None:
        # 计算目标点在相机坐标系下的三维坐标
        object_point = calculate_3d_coordinates(laser_x, laser_y, camera_matrix, laser_plane_distance)

        print("目标点在相机坐标系下的三维坐标:", object_point)
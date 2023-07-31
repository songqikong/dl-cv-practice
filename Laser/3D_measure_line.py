import numpy as np
import cv2

def calibrate_laser_plane(image_path, chessboard_size, square_size, camera_matrix, dist_coeffs):
    # 读取图像
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
    if ret == False:
        print("找不到角点")
        return

    # 获取棋盘格角点的世界坐标
    obj_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    if ret:
        # 标定相机，获得相机的畸变系数和旋转矩阵、平移矩阵
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera([obj_points], [corners], gray_image.shape[::-1], None, None)

    # 使用solvePnP获得激光平面参数
    _, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)
    rmat, _ = cv2.Rodrigues(rvec)
    laser_plane_params = np.dot(rmat.T, tvec)

    return laser_plane_params

# def calculate_3d_coordinates(laser_line_start, laser_line_end, camera_matrix, laser_plane_params):
#     # 构建激光线在图像上的端点坐标数组
#     laser_line_points = np.column_stack((laser_line_start, laser_line_end))
#
#     # 使用undistortPoints函数将像素坐标转换为归一化坐标
#     normalized_coordinates = cv2.undistortPoints(laser_line_points.reshape(-1, 1, 2), camera_matrix, None, None)
#
#     # 将归一化坐标转换为相机坐标系下的三维坐标
#     ones_column = np.ones((normalized_coordinates.shape[0], 1), dtype=np.float32)
#     camera_coordinates = np.hstack((normalized_coordinates[:, 0, :], ones_column))
#
#     # 计算激光线在相机坐标系下的三维坐标
#     laser_plane_distance = -laser_plane_params[2] / np.dot(laser_plane_params[:2], camera_coordinates.T)
#     camera_coordinates_3d = np.multiply(camera_coordinates, laser_plane_distance[:, None])
#
#     return camera_coordinates_3d
def calculate_3d_coordinates(laser_line_start, laser_line_end, camera_matrix, laser_plane_params):
    # 构建激光线在图像上的端点坐标数组
    laser_line_points = np.array([laser_line_start, laser_line_end], dtype=np.float32)

    # 将像素坐标转换为相机坐标系下的三维坐标
    camera_coordinates_3d = cv2.triangulatePoints(camera_matrix, np.eye(3), laser_line_points.T, np.zeros((4, 1)))

    # 归一化，即将齐次坐标转换为非齐次坐标
    camera_coordinates_3d /= camera_coordinates_3d[3]

    # 由于OpenCV中的三维坐标为齐次坐标，我们只需要前三个元素表示的点的三维坐标
    camera_coordinates_3d = camera_coordinates_3d[:3].T

    # 计算激光线在相机坐标系下的三维坐标
    laser_plane_distance = -laser_plane_params[3] / np.dot(laser_plane_params[:3], camera_coordinates_3d.T)
    laser_line_3d = np.multiply(camera_coordinates_3d, laser_plane_distance[:, None])

    return laser_line_3d


def find_laser_line_endpoints(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 对图像进行边缘检测
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # 使用HoughLines函数检测图像中的直线
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

    # 获取激光线的端点坐标
    laser_line_endpoints = []
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        laser_line_endpoints.append([(x1, y1), (x2, y2)])

    return laser_line_endpoints


if __name__ == "__main__":
    # 相机内参矩阵，需要根据实际相机进行设置
    camera_matrix = np.array([[496.57849394, 0, 278.74782676],
                              [0, 510.37523816, 264.13207105],
                              [0, 0, 1]], dtype=np.float32)

    # 畸变系数，设置为空数组
    dist_coeffs = np.array([])

    # 棋盘格大小
    chessboard_size = (8, 6)  # 根据实际情况设置
    square_size = 0.02  # 根据实际情况设置，单位为米

    image_path = "./16.bmp"  # 根据实际情况设置图片路径
    laser_plane_params = calibrate_laser_plane(image_path, chessboard_size, square_size, camera_matrix, dist_coeffs)

    # 查找激光线的起点和终点坐标
    laser_line_endpoints = find_laser_line_endpoints(image_path)
    # 激光线在图像上的端点坐标，根据实际情况设置
    laser_line_start = laser_line_endpoints[0]
    laser_line_end = laser_line_endpoints[1]

    # 计算激光线在相机坐标系下的三维坐标
    camera_coordinates_3d = calculate_3d_coordinates(laser_line_start, laser_line_end, camera_matrix, laser_plane_params)

    print("激光线在相机坐标系下的三维坐标:", camera_coordinates_3d)
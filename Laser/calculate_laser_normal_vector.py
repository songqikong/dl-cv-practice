import numpy as np
import cv2

def calibrate_laser_plane(image_path, chessboard_size, square_size, camera_matrix):
    # 读取图片
    img = cv2.imread(image_path)
    cv2.imshow('2', img)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('1', gray_img)

    # 寻找棋盘角点
    ret, corners = cv2.findChessboardCorners(gray_img, chessboard_size,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        # 寻找棋盘的亚像素角点
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray_img, corners, (5, 5), (-1, -1), criteria)

        # 绘制角点
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 获取棋盘点的三维坐标
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        # 转换为世界坐标系
        world_points = objp

        # 转换为相机坐标系
        image_points = corners.squeeze()

        # 使用solvePnP函数来计算外参矩阵
        _, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix, None)

        # 从旋转向量计算旋转矩阵
        rmat, _ = cv2.Rodrigues(rvec)

        # 由于棋盘在世界坐标系中位于xy平面上，所以其法向量为(0, 0, 1)
        normal_vector = np.array([0, 0, 1], dtype=np.float32)

        # 计算激光平面参数
        laser_plane_params = np.dot(rmat.T, normal_vector)

        # 激光平面参数归一化
        laser_plane_params /= np.linalg.norm(laser_plane_params)

        return laser_plane_params

    else:
        print("未找到棋盘角点。")

def compute_laser_normal_vector(rvec, rmat):
    # 棋盘在世界坐标系中位于xy平面上，所以其法向量为(0, 0, 1)
    world_normal_vector = np.array([0, 0, 1], dtype=np.float32)

    # 将法向量从世界坐标系转换到相机坐标系
    camera_normal_vector = np.dot(rmat.T, world_normal_vector)

    # 法向量归一化
    laser_normal_vector = camera_normal_vector / np.linalg.norm(camera_normal_vector)

    return laser_normal_vector

if __name__ == "__main__":
    # 相机内参矩阵，需要根据实际相机进行设置
    camera_matrix = np.array([[496.57849394, 0, 278.74782676],
                              [0, 510.37523816, 264.13207105],
                              [0, 0, 1]], dtype=np.float32)

    # 棋盘格大小
    chessboard_size = (8, 6)  # 根据实际情况设置
    square_size = 0.02  # 根据实际情况设置，单位为米

    image_path = "./16.bmp"  # 根据实际情况设置图片路径
    ret, rvec, tvec = calibrate_laser_plane(image_path, chessboard_size, square_size, camera_matrix)

    if ret:
        # 从旋转向量计算旋转矩阵
        rmat, _ = cv2.Rodrigues(rvec)

        # 计算激光平面法向量
        laser_normal_vector = compute_laser_normal_vector(rvec, rmat)

        print("激光平面法向量:", laser_normal_vector)
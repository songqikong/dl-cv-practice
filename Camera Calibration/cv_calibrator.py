import cv2
import numpy as np
import glob

# 定义棋盘的维度
CHECKBOARD = (8, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 为每个图像创建3D和3D坐标存储向量
objpoints = []
imgpoints = []

# 为3D点定义世界坐标
objp = np.zeros((1, CHECKBOARD[0] * CHECKBOARD[1], 3)
                , np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKBOARD[0],
                 0: CHECKBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# 读取图片
images = glob.glob('../Laser/img/*.bmp')
for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # cv2.imshow('img', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 开始寻找棋盘点
    # 如果在图片中找到了确定好的期盼点数目，则ren=true
    # flag = cv2.CALIB_CB_ADAPTIVE_THRESH
    flag = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

    ret, corners = cv2.findChessboardCorners(gray,
                                             CHECKBOARD,
                                             flag)
    # 如果找到了所有期盼点，就确定坐标并在棋盘上标记出来
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,
                                    corners,
                                    (11, 11),
                                    (-1, -1),
                                    criteria)

        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img,
                                        CHECKBOARD,
                                        corners2,
                                        ret)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # cv2.destroyWindow()

    # h, w = img.shape[:2]

    # 根据已知的点和期盼点的位置进行相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.savez('B.npz', ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

img = cv2.imread('new.jpg')
img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

h, w = img.shape[:2]

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('calibresult.png', dst)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

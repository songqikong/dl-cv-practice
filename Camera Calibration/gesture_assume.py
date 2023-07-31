import numpy as np
import cv2 as cv
import glob

# 读取事先存好的数据
with np.load('B.npz') as X:
    for i in X:
        print(i)
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
CHECKBOARD = (9, 6)

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 5)
    return img


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((1, CHECKBOARD[0] * CHECKBOARD[1], 3)
                , np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKBOARD[0],
                 0: CHECKBOARD[1]].T.reshape(-1, 2)

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

for fname in glob.glob('D:\\Archive of Code\\Pytorch\\Camera Calibration\\calibresult.png'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, CHECKBOARD, None)

    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 找到旋转和平移向量
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        # 投射 3D 点到平面图像上
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img, corners2, imgpts)
        cv.imshow('img', img)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite(fname[:6] + '.png', img)

cv.destroyAllWindows()

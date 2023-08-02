import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
from scipy import signal

'''
    提取激光图像的像素中心坐标
    使用角点检测失败：实验证明单纯的角点检测不能得到中心点坐标，即使提取到中心的角点，精度也比较差

    使用霍夫变换，检测直线，进而计算两个直线的交点
'''

# 输入灰度图
src = cv.imread('./Photo/1.jpg')
gray_src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
# 将图像转换为浮点型
# gray_src = np.float32(gray_src)
# 高斯滤波
filter_src = cv.GaussianBlur(gray_src, (7, 7), 1.5)

# 计算频率直方图 用于观察图像特征
hist = cv.calcHist([filter_src], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.show()

# 固定阈值130 实验得出的
ret, bin_img = cv.threshold(filter_src, 140, 255, cv.THRESH_BINARY)

# 开操作
open_img = cv.morphologyEx(bin_img, cv.MORPH_OPEN, (5, 5), 10)

# Harris 角点检测
dst = cv.cornerHarris(open_img, 2, 3, 0.04)
# 突出角点
# dst = cv.dilate(dst, None)
src[dst > 0.01 * dst.max()] = [0, 0, 255]
cv.imshow('cornerHarris', src)
cv.waitKey(0)
cv.imshow('dst', dst)
cv.waitKey(0)

# Shi-Tomasi 检测
# corners = cv.goodFeaturesToTrack(open_img, 1, 0.01, 10)
# corners_array = np.int0(corners)
#
# for i in corners:
#     x, y = i.ravel()
#     cv.circle(src, (int(x), int(y)), 3, 255, -1)
#
# cv.imshow('src', src)
# cv.waitKey(0)


# open_img[dst > 0.01 * dst.max()] = [0, 0, 255]

# cv.imshow('bin', open_img)
# cv.waitKey(0)
# cv.imshow('dst', dst)
# cv.waitKey(0)

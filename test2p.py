import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
from scipy import signal

'''
    提取激光图像的像素中心坐标
    使用角点检测失败：实验证明单纯的角点检测不能得到中心点坐标，即使提取到中心的角点，精度也比较差

    先边缘检测，然后使用霍夫变换，检测直线，进而计算两个直线的交点

    从Canny入手， 或者，使用mask
'''

# 输入灰度图
src = cv.imread('./Photo/5.jpg')
h, w, c = src.shape
gray_src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)

# # Cany 边缘检测
edge = cv.Canny(gray_src, 130, 142)
cv.imshow('Canny', edge)
cv.waitKey(0)

# 霍夫变换
lines = cv.HoughLinesP(edge, 0.45, np.pi/180, 100, minLineLength=50, maxLineGap=10)
print(lines)
for i in range(len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        cv.line(src, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv.imshow('result', src)
cv.waitKey(0)


# # Cany 边缘检测
# edge = cv.Canny(filter_src, 130, 142)
# cv.imshow('Canny', edge)
# cv.waitKey(0)




# # 霍夫变换
# # lines = cv.HoughLinesP(open_img, 1.0, np.pi/180, 100, 100, 10)
# lines = cv.HoughLines(edge, 0.5, np.pi / 180, 100)
# print(lines)
# # x1, y1 = lines[1]
#
# result_line_array = []
# for i in range(len(lines)):
#     for rho, theta in lines[i]:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + w * (-b))
#         y1 = int(y0 + w * a)
#         x2 = int(x0 - w * (-b))
#         y2 = int(y0 - w * a)
#
#         cv.line(src, (x1, y1), (x2, y2), (0, 0, 255), 2)
# cv.imshow('result', src)
# cv.waitKey(0)


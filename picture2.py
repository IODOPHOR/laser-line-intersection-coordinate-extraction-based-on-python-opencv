import cv2.cv2 as cv
import numpy as np
import matplotlib.pylab as plt
from scipy import signal

'''

    图片8
    添加滤波以制左下角的噪声

'''


def hough2les(rho, theta, w):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + w * (-b))
    y1 = int(y0 + w * a)
    x2 = int(x0 - w * (-b))
    y2 = int(y0 - w * a)

    k = float(y2 - y1) / float(x2 - x1)
    b = (-1) * x1 * k + y1
    return k, b


src = cv.imread('./Photo/2.jpg')
high, width, channel = src.shape
hs_img = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
# cv.imshow('hs', hs_img)
# cv.waitKey(0)

# 高斯滤波
filter_src = cv.GaussianBlur(hs_img, (7, 7), 1.5)
# 固定阈值144 实验得出的
ret, bin_img = cv.threshold(filter_src, 144, 255, cv.THRESH_BINARY)
# 开操作
open_img = cv.morphologyEx(bin_img, cv.MORPH_OPEN, (5, 5), 10)
cv.imshow('open_img', open_img)
cv.waitKey(0)


# edge_g_img = cv.Canny(Gaus_img, 130, 140)
# cv.imshow('edge_g_img', edge_g_img)
# cv.waitKey(0)
# edge_img = edge_g_img

edge_img = cv.Canny(open_img, 130, 142)
cv.imshow('edge_img', edge_img)
cv.waitKey(0)



# 霍夫变换
# p 的间隔为0.45 角度间隔为 1度， 长度大于100才被认为是直线
lines = cv.HoughLines(edge_img, 0.55, np.pi / 120, 100)
# print(lines)

show_img = np.zeros((high, width), np.uint8)

# result_line_array = []
for i in range(len(lines)):
    for rho, theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + width * (-b))
        y1 = int(y0 + width * a)
        x2 = int(x0 - width * (-b))
        y2 = int(y0 - width * a)

        cv.line(show_img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv.line(src, (x1, y1), (x2, y2), (0, 0, 255), 1)
cv.imshow('show_img', src)
cv.waitKey(0)
# exit()
# 角点检测
# 检测四个点，0.01是品质因数0.1-0.01之间， 10是两个点之间的最小距离
corners = cv.goodFeaturesToTrack(show_img, 4, 0.01, 1)
corners_array = np.int0(corners)

x_value = 0.0
y_value = 0.0
for i in corners:
    x, y = i.ravel()
    x_value += x
    y_value += y
    # cv.circle(src, (int(x_value), int(y_value)), 3, 100, -1)

x_value /= len(corners)
y_value /= len(corners)
cv.circle(src, (int(x_value), int(y_value)), 3, 255, -1)
print('交点:{}, {}\n'.format(x_value, y_value))
cv.imshow('result', src)
cv.waitKey(0)

exit()

# import cv2 as cv
# import numpy as np
# import matplotlib.pylab as plt
# from scipy import signal
#
# '''
#     提取激光图像的像素中心坐标
#     使用角点检测失败：实验证明单纯的角点检测不能得到中心点坐标，即使提取到中心的角点，精度也比较差
#
#     先边缘检测，然后使用霍夫变换，检测直线，进而计算两个直线的交点
#
#     图片1
# '''
#
# # 输入灰度图
# src = cv.imread('./Photo/2.jpg')
# h, w, c = src.shape
# gray_src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
# # 高斯滤波
# filter_src = cv.GaussianBlur(gray_src, (7, 7), 1.5)
# # 固定阈值144 实验得出的
# ret, bin_img = cv.threshold(filter_src, 144, 255, cv.THRESH_BINARY)
# # 开操作
# open_img = cv.morphologyEx(bin_img, cv.MORPH_OPEN, (5, 5), 10)
# cv.imshow('open_img', open_img)
# cv.waitKey(0)
#
# # # Cany 边缘检测
# edge = cv.Canny(open_img, 130, 142)
# cv.imshow('Canny', edge)
# cv.waitKey(0)
# # exit()
# # 霍夫变换
# # lines = cv.HoughLinesP(open_img, 1.0, np.pi/180, 100, 100, 10)
# lines = cv.HoughLines(edge, 0.5, np.pi / 180, 100)
# print(lines)
# # x1, y1 = lines[1]
#
# result_line_array = []
# for i in [0, 2]:
#     for rho1, theta1 in lines[i]:
#         rho2 = lines[i + 1][0][0]
#         theta2 = lines[i + 1][0][1]
#         rho = (rho2 + rho1) / 2.0
#         theta = (theta1 + theta2) / 2.0
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + w * (-b))
#         y1 = int(y0 + w * a)
#         x2 = int(x0 - w * (-b))
#         y2 = int(y0 - w * a)
#         k = float(y2 - y1) / float(x2 - x1)
#         b = -x1 * k + y1
#         result_line_array.append(k)
#         result_line_array.append(b)
#         cv.line(src, (x1, y1), (x2, y2), (0, 0, 255), 2)
# k1 = result_line_array[0]
# b1 = result_line_array[1]
# k2 = result_line_array[2]
# b2 = result_line_array[3]
#
# x = float(b2 - b1) / float(k1 - k2)
# y = k1 * x + b1
# print('{}, {}'.format(x, y))
# cv.circle(src, (int(x), int(y)), 10, (0, 238, 0))
#
# cv.imshow('result', src)
# cv.waitKey(0)
#
# exit()

# 霍夫变换
# lines = cv.HoughLinesP(open_img, 1.0, np.pi/180, 100, 100, 10)
# print(lines)
# for i in range(len(lines)):
#     for x1,y1,x2,y2 in lines[i]:
#         cv.line(src, (x1,y1), (x2,y2), (0, 0, 255), 2)
# cv.imshow('result', src)
# cv.waitKey(0)


# # Cany 边缘检测
# edge = cv.Canny(filter_src, 130, 142)
# cv.imshow('Canny', edge)
# cv.waitKey(0)




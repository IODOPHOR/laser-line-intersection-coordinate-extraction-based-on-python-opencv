import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
from scipy import signal

'''
    
    图片6
    改进：不算坐标了，直接使用角点检测
    
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
    b = (-1)*x1*k + y1
    return k, b


src = cv.imread('./Photo/6.jpg', cv.IMREAD_COLOR)
high, width, channel = src.shape
hs_img = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
# cv.imshow('hs', hs_img)
# cv.waitKey(0)

# # 高斯滤波
# Gaus_img = cv.GaussianBlur(hs_img, (7, 7), 1.5)
# # cv.imshow('gauss', hs_img)
# # cv.waitKey(0)
#
# # Canny边缘检测
# mask_img = cv.Canny(Gaus_img, 130, 142)
# cv.imshow('edge', mask_img)
# cv.waitKey(0)
#
# elememt_1 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
#
# dilation_img = cv.dilate(mask_img, elememt_1, iterations=10)
#
# cv.imshow('di_img', dilation_img)
# cv.waitKey(0)

edge_img = cv.Canny(hs_img, 130, 140)
cv.imshow('edge_img', edge_img)
cv.waitKey(0)


# for i in range(high):
#     for j in range(width):
#         if dilation_img[i][j] == 255:
#             edge_img[i][j] = 0

cv.imshow('edge_1_img', edge_img)
cv.waitKey(0)

# 霍夫变换
# p 的间隔为0.45 角度间隔为 1度， 长度大于100才被认为是直线
lines = cv.HoughLines(edge_img, 0.45, np.pi / 180, 100)
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
    # print('交点:{},{}\n'.format(x, y))

x_value /= len(corners)
y_value /= len(corners)
cv.circle(src, (int(x_value), int(y_value)), 3, 255, -1)
print('交点:{},{}\n'.format(x_value, y_value))
cv.imshow('result', src)
cv.waitKey(0)

exit()

'''
    以下代码是针对Hough变换得到的直线进行处理，先将统一方向上的直线平均
    然后在，算交点

    但！ 所做的处理只适用于这张图片不具有普适性


'''


#
# # 检测交点
# '''
#     Hough 变换得到多条曲线
#     将 角度相差小于0.1 的直线求平均，认为是同一条直线
# '''
# # 确定两条直线
# flag_value = 1
# camp_theta_value = lines[0][0][1]
# theta_1_sum = lines[0][0][1]
# rho_1_sum = lines[0][0][0]
# theta_2_value = 0.0
# rho_2_value = 0.0
# '''
#     明确一点：霍夫变换得到的直线一定没有识别错误，所以，得到的直线数组只可能有两个斜率范围
#     1、建立中转数组
#     2、将同一个斜率范围内的直线平均
#     3、计算交点即可
#
#     注：斜率为计算时除数为0的情况 ！！！
#
#     不算了，直接角点检测即可
# '''
#
# '''
# 最好先排序
# '''
#
#
# for i in range(1, len(lines)):
#     for rho, theta in lines[i]:
#         if np.abs(theta - camp_theta_value) <= 1:
#             flag_value += 1
#             theta_1_sum += theta
#             rho_1_sum += rho
#         else:
#             theta_2_value = theta
#             rho_2_value = rho
#
#
#
# # 求平均
# theta_1_value = theta_1_sum / flag_value
# rho_1_value = rho_1_sum / flag_value
#
# line_1_k, line_1_b = hough2les(rho_1_value, theta_1_value, width)
# line_2_k, line_2_b = hough2les(rho_2_value, theta_2_value, width)
#
# point_x = int((line_2_b-line_1_b)/(line_1_k-line_2_k))
# point_y = int(line_1_k*point_x + line_1_b)
#
#
# print('{}, {}'.format(point_x, point_x))
# cv.circle(src, (point_x, point_y), 10, (0, 238, 0))
#
# cv.imshow('result', src)
# cv.waitKey(0)
#
# exit()

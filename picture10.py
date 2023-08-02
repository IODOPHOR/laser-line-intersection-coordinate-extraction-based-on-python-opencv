import cv2 as cv
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


src = cv.imread('./Photo/10.jpg', cv.IMREAD_COLOR)
high, width, channel = src.shape
hs_img = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
# cv.imshow('hs', hs_img)
# cv.waitKey(0)

Gaus_img = cv.GaussianBlur(hs_img, (3, 3), 1.5)
cv.imshow('gauss', hs_img)
cv.waitKey(0)

edge_g_img = cv.Canny(Gaus_img, 130, 140)
cv.imshow('edge_g_img', edge_g_img)
cv.waitKey(0)
edge_img = edge_g_img

# edge_img = cv.Canny(hs_img, 130, 140)
# cv.imshow('edge_img', edge_img)
# cv.waitKey(0)



# 霍夫变换
# p 的间隔为0.45 角度间隔为 1度， 长度大于100才被认为是直线
lines = cv.HoughLines(edge_img, 0.5, np.pi / 120, 100)
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
        cv.line(src, (x1, y1), (x2, y2), (0, 255, 20), 1)
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

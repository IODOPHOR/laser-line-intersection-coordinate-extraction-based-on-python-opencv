import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
from scipy import signal

''' 
    提取激光图像的像素中心坐标
    使用角点检测失败：实验证明单纯的角点检测不能得到中心点坐标，即使提取到中心的角点，精度也比较差

    先边缘检测，然后使用霍夫变换，检测直线，进而计算两个直线的交点
    
    Canny检测会检测到文字， 考虑使用其他的检测方法，
    
    彩色图像先滤波，然后二值化，在边缘检测，太过于依赖二值化的阈值，阈值的设定直接决定图像分割的好坏
    本题，文字太突出了，不好做二值化，本来想提取颜色分量，但效果不是很好，相减后的结果并不能完全消除噪声
    
    效果差： 考虑使用Sobel算子，提取水平、竖直方向的特征、 效果不是很好
    
    考虑腐蚀，构造结构元素
    
'''

# 输入灰度图
src = cv.imread('./Photo/5.jpg')
cv.imshow('src', src)
cv.waitKey(0)
h, w, c = src.shape
gray_src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
# 高斯滤波
filter_src = cv.GaussianBlur(gray_src, (3, 3), 1.5)
# cv.imshow('filter', filter_src)
# cv.waitKey(0)

# 计算频率直方图 用于观察图像特征
hist = cv.calcHist([filter_src], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.show()

# 固定阈值140 实验得出的
ret, bin_img = cv.threshold(filter_src, 150, 255, cv.THRESH_BINARY)
# 开操作
# open_img = cv.morphologyEx(bin_img, cv.MORPH_OPEN, (5, 5), 10)
# cv.imshow('open_img', open_img)
# cv.waitKey(0)

# # Cany 边缘检测

edge_x = cv.Sobel(gray_src, cv.CV_16S, 0, 1)
edge_y = cv.Sobel(gray_src, cv.CV_16S, 1, 0)

abs_x = cv.convertScaleAbs(edge_x)
abs_y = cv.convertScaleAbs(edge_y)

cv.imshow('abs_x', abs_x)
cv.waitKey(0)
cv.imshow('abs_y', abs_y)
cv.waitKey(0)





edge = cv.Canny(gray_src, 130, 142)
cv.imshow('Canny', edge)
cv.waitKey(0)

open_img = cv.morphologyEx(edge, cv.MORPH_OPEN, (5, 5), 10)
cv.imshow('open_img', open_img)
cv.waitKey(0)

# exit()

# 霍夫变换
# lines = cv.HoughLinesP(open_img, 1.0, np.pi/180, 100, 100, 10)

lines = cv.HoughLines(edge, 0.45, np.pi / 180, 100)

print(lines)
num, width, channel = lines.shape
middle_array = np.zeros((num, 2), np.float)
# 画检测出的线段
for i in range(len(lines)):
    for rho, theta in lines[i]:
        middle_array[i][0] = rho
        middle_array[i][1] = theta

head = 0
rear = -1

head_value = middle_array[0][1]
for i in range(len(middle_array)):
    if head_value == middle_array[i][1]:
        rear = i


for i in range(rear+1, len(middle_array)):
    for rho, theta in lines[i]:
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + w * (-b))
        y1 = int(y0 + w * a)
        x2 = int(x0 - w * (-b))
        y2 = int(y0 - w * a)

        cv.line(src, (x1, y1), (x2, y2), (0, 238, 0), 2)

cv.imshow('result', src)
cv.waitKey(0)

exit()

result_line_array = []
# 求 相交线
for i in range(len(lines) - 1):
    for rho, theta in lines[i]:
        rho1 = lines[i + 1][0][0]
        theta1 = lines[i + 1][0][1]
        if theta == theta1:
            continue
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + w * (-b))
        y1 = int(y0 + w * a)
        x2 = int(x0 - w * (-b))
        y2 = int(y0 - w * a)

        # cv.line(src, (x1, y1), (x2, y2), (0, 238, 0), 2)






for i in [0, 2]:
    for rho1, theta1 in lines[i]:
        rho2 = lines[i + 1][0][0]
        theta2 = lines[i + 1][0][1]
        rho = (rho2 + rho1) / 2.0
        theta = (theta1 + theta2) / 2.0
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + w * (-b))
        y1 = int(y0 + w * a)
        x2 = int(x0 - w * (-b))
        y2 = int(y0 - w * a)
        k = float(y2 - y1) / float(x2 - x1)
        b = -x1 * k + y1
        result_line_array.append(k)
        result_line_array.append(b)

k1 = result_line_array[0]
b1 = result_line_array[1]
k2 = result_line_array[2]
b2 = result_line_array[3]

x = float(b2 - b1) / float(k1 - k2)
y = k1 * x + b1
print('{}, {}'.format(x, y))
cv.circle(src, (int(x), int(y)), 10, (0, 0, 0))

cv.imshow('result', src)
cv.waitKey(0)

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




import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 输入灰度图
src = cv.imread('./Photo/5.jpg', cv.IMREAD_COLOR)
hs_img = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
cv.imshow('hs', hs_img)
cv.waitKey(0)

# ret, bin_img = cv.threshold(hs_img, 160, 255, cv.THRESH_BINARY)
# ret, b2_img = cv.threshold(hs_img, 100, 255, cv.THRESH_BINARY)
# cv.imshow('bin_img', bin_img)
# cv.waitKey(0)
# cv.imshow('b2_img', b2_img)
# cv.waitKey(0)
#
# sub_img = cv.subtract(b2_img, bin_img)
# cv.imshow('sub', sub_img)
# cv.waitKey(0)


# hsv_img = cv.cvtColor(src, cv.COLOR_RGB2HSV)
# cv.imshow('hsv', hsv_img)
# cv.waitKey(0)


high, width, channel = src.shape
w = width
# gray_img = np.zeros((high, width), np.uint8)
# # (H, S, V) = cv.split(src)
# (B, G, R) = cv.split(src)
#
# for i in range(high):
#     for j in range(width):
#         g = B[i][j]
#         b = G[i][j]
#         r = R[i][j]
#         if (180 <= r <= 255) and (180 <= b <= 255) and (180 <= g <= 255):
#             gray_img[i][j] = hs_img[i][j]

# sub_img = cv.subtract(hs_img, gray_img)
# cv.imshow('sub', sub_img)
# cv.waitKey(0)


edge = cv.Canny(hs_img, 130, 142)
cv.imshow('Canny', edge)
cv.waitKey(0)


lines = cv.HoughLines(edge, 0.6, np.pi / 90, 100)
print(lines)
# x1, y1 = lines[1]

result_line_array = []
for i in range(len(lines)):
    for rho, theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + w * (-b))
        y1 = int(y0 + w * a)
        x2 = int(x0 - w * (-b))
        y2 = int(y0 - w * a)

        cv.line(src, (x1, y1), (x2, y2), (0, 0, 255), 2)

# open_img = cv.morphologyEx(gray_img, cv.MORPH_OPEN, (5, 5), 10)
# cv.imshow('result', open_img)
# cv.waitKey(0)




cv.imshow('gray_img', src)
cv.waitKey(0)




# (B, G, R) = cv.split(src)
# cv.imshow('R', R)
# cv.waitKey(0)
exit()
gray_src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
# 高斯滤波
filter_src = cv.GaussianBlur(gray_src, (7, 7), 1.5)

# 计算频率直方图 用于观察图像特征
hist = cv.calcHist([filter_src], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()


# # 提取红色
# for i in range(high):
#     for j in range(width):
#         h = H[i][j]
#         s = S[i][j]
#         v = V[i][j]
#         if (0 < h < 10) or (156 < h < 180):
#             if 43 < s <= 255:
#                 if 46 < v <= 255:
#                     gray_img[i][j] = 255

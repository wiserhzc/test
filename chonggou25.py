import numpy as np
import cv2
import os
from sklearn import linear_model
import cvxpy as cvx
from PIL import Image
import matplotlib.pyplot as plt
a = np.array([])
file_pathname = r'C:\Users\Admin\Desktop\data12\1'
print(os.listdir(file_pathname))
for filename in os.listdir(file_pathname):
    img = cv2.imread(file_pathname + '/' + filename, 0)
    img1 = img.ravel()
    a = np.append(a, img1)
a = a.reshape(50, 307200)
y = a
t = np.loadtxt(r'C:\Users\Admin\Desktop\data12\y考虑响应切片间隔10.csv', delimiter=',')
# t = 3*t
x11 = np.loadtxt(r'C:\Users\Admin\Desktop\data12\x切片间隔10.csv', delimiter=',')
a1 = np.linalg.pinv(t)
x = np.dot(a1, y)

save_out = r'C:\Users\Admin\Desktop\data12\tu25/'
i = 0
# while i < 25:
#     b = x[i, :]
#     bmin = min(b)  # 归一化255
#     bm = b - bmin
#     bmax = max(bm)
#     b1 = 255 * (bm / bmax)
#     b1 = b1.reshape(480, 640)
#     save_path = save_out + str(int(x11[i])) + 'nm.png'
#     cv2.imwrite(save_path, b1)
#     i += 1
    # b = b.astype(np.int16)
    # b = b.reshape(480, 640)
    # save_path = save_out + str(int(x11[i])) + 'nm.png'
    # cv2.imwrite(save_path, b)
    # i += 1
clf = linear_model.Ridge(fit_intercept=False)
clf.set_params(alpha=0.56)
clf.fit(t, y)
x1=np.array(clf.coef_)
print(x1.shape)
while i < 25:

    b = x1[:, i]
    b = b.reshape(480, 640)
    # c1 = np.mean(b[300:340,90:130])
    # b[b > c1] = c1
    # b[b > 400] = 400
    # b[b < -(0.15*b11)] = -(0.15*b11)
    bmin = np.min(b)  # 归一化255
    bm = b - bmin
    bmax = np.max(bm)
    b1 = 255 * (bm / bmax)
    b1 = b1.astype(np.uint8)
    save_path = save_out + str(int(x11[i])) + 'nm.png'
    cv2.imwrite(save_path, b1)
    i += 1





    #
    # bmin = min(b)  # 归一化255
    # bm = b - bmin
    # bmax = max(bm)
    # b1 = 255 * (bm / bmax)
    # b1 = b1.reshape(480, 640)
    # b1 = b1.astype(np.uint8)
    # save_path = save_out + str(int(x11[i])) + 'nm.png'
    # cv2.imwrite(save_path, b1)
    # i += 1

    # b = b.astype(np.uint8)
    # b = b.reshape(480, 640)
    # save_path = save_out + str(int(x11[i])) + 'nm.png'
    #
    # cv2.imwrite(save_path, b)
    # i += 1
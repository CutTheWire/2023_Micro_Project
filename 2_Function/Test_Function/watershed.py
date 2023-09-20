import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("C:/Users/sjmbe/TW/TEST/230908/161906_Micro.jpg")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

cv2.imshow("",opening)
cv2.waitKey(0)
# # sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=4)

# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.38*dist_transform.max(),255,0)

# # Finding unknown region
# sure_fg = np.uint8(sure_fg)

# plt.imshow(sure_fg, cmap='viridis')
# plt.colorbar()
# plt.title('Distance Transform of Sure Foreground')
# plt.show()

# 이를 입력으로 거리 변환을 수행합니다.
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

# 거리 변환 결과를 확인하기 위해 그래프로 표시할 수 있습니다.
plt.imshow(dist_transform, cmap='viridis')
plt.colorbar()
plt.title('Distance Transform of Sure Foreground')
plt.show()

# cv2.imshow("e", sure_fg)
# cv2.waitKey(0)

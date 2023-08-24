import cv2
import numpy as np
from IPython.display import Image, display
from matplotlib import pyplot as plt

# 이미지 표시 함수
def imshow(img, ax=None):
    if ax is None:
        ret, encoded = cv2.imencode(".jpg", img)
        display(Image(encoded))
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')

# 이미지 로드
img = cv2.imread(r"/content/drive/MyDrive/tinyparts_images/WIN_20230808_09_54_54_Pro.jpg")

# 이미지 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 이미지 표시
imshow(img)

# 노이즈 제거
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bin_img = cv2.morphologyEx(bin_img,
                           cv2.MORPH_OPEN,
                           kernel,
                           iterations=2)
imshow(bin_img)

# 1행 2열의 서브플롯 생성
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
# 배경 영역 확장
sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
imshow(sure_bg, axes[0,0])
axes[0, 0].set_title('Sure Background')

# 거리 변환
dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
imshow(dist, axes[0,1])
axes[0, 1].set_title('Distance Transform')

# 전경 영역
ret, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, cv2.THRESH_BINARY)
sure_fg = sure_fg.astype(np.uint8)
imshow(sure_fg, axes[1,0])
axes[1, 0].set_title('Sure Foreground')

# 알 수 없는 영역
unknown = cv2.subtract(sure_bg, sure_fg)
imshow(unknown, axes[1,1])
axes[1, 1].set_title('Unknown')

plt.show()

# 마커 라벨링
# 확실한 전경
ret, markers = cv2.connectedComponents(sure_fg)

# 모든 라벨에 1을 더하여 배경이 0이 아니라 1이 되도록 함
markers += 1
# 알 수 없는 영역을 0으로 표시
markers[unknown == 255] = 0

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(markers, cmap="tab20b")
ax.axis('off')
plt.show()

# 워터셰드 알고리즘
path = '/content/drive/MyDrive/Output/'
markers = cv2.watershed(img, markers)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(markers, cmap="tab20b")
ax.axis('off')
plt.show()

labels = np.unique(markers)

coins = []
count = 0
areas = []
for label in labels[2:]:
    # 라벨 영역만 전경으로, 나머지는 배경으로 하는 이진 이미지 생성
    target = np.where(markers == label, 255, 0).astype(np.uint8)
    # 생성된 이진 이미지에 대해 윤곽 추출 수행
    contours, hierarchy = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    val = cv2.contourArea(contours[0])
    print(val)
    areas.append(val)
    coins.append(contours[0])
    count = count + 1
    # 윤곽 그리기

image = cv2.drawContours(img, coins, -1, color=(0, 23, 223), thickness=1)
imshow(image)
cv2.imwrite("output.jpg", image)

print(areas)

# 이상치 제거
# 1000보다 큰 이상치 제거
mean_finder = []
for i in range(len(areas)):
    if areas[i] < 1000:
        mean_finder.append(areas[i])

print(mean_finder)

import statistics
mean_size = statistics.mean(mean_finder)

param = 0.90
threshold_mean = .90 * mean_size

print(threshold_mean)

size = len(areas)
for i in range(size):
    if areas[i] < threshold_mean:
        areas.pop(i)
        coins.pop(i)
        print("Deleted")
        size = size - 1

counter = 0
contour_with_more_area_than_normal = []
for i, area in enumerate(areas):
    res = area // threshold_mean
    if res > 1:
        print(area)
        contour_with_more_area_than_normal.append(coins[i])
    counter = counter + res

high_draw = cv2.drawContours(image, contour_with_more_area_than_normal, -1, color=(255,255,0), thickness=3)

cv2.imwrite("Contour Area More than Normal.jpg", high_draw)

# 종료

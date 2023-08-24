import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt

def f_contours(frame):
    # Canny 엣지 검출
    edges = cv2.Canny(frame, threshold1=30, threshold2=100)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# 이미지를 로드하고 그레이스케일로 변환 154637, 154636, 152636
image = cv2.imread('2023-08-23_154637.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 가우시안 블러를 적용하여 노이즈 제거
blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# 원본 이미지 복사
image_copy = image.copy()

# 윤곽선 내부 영역을 채워서 마스크 생성
mask = np.zeros_like(blurred)

contours = f_contours(blurred)

cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

# 마스크를 사용하여 윤곽선 내부 영역만 추출
masked_image = cv2.bitwise_and(image_copy, image_copy, mask=mask)
binary_img = copy.deepcopy(masked_image)

# RGB 채널 데이터 추출
r_channel_data = binary_img[:, :, 2]
b_channel_data = binary_img[:, :, 0]

# Find the most frequent pixel value in the R channel
hist, bins = np.histogram(r_channel_data.flatten(), bins=range(1, 102))  # 1~101 값 범위
r_most_frequent_value = np.argmax(hist) + 1  # +1은 bin의 시작 값이 1임을 보정

hist, bins = np.histogram(b_channel_data.flatten(), bins=range(1, 102))  # 1~101 값 범위
b_most_frequent_value = np.argmax(hist) + 1  # +1은 bin의 시작 값이 1임을 보정

print(f"The most frequent R channel value is: {r_most_frequent_value}")
print(f"The most frequent B channel value is: {b_most_frequent_value}")

# Apply condition to modify pixel values
condition = binary_img[:, :, 0] <= int(b_most_frequent_value*1.50)  # channel values less than or equal
condition2 = binary_img[:, :, 2] == 0 # channel values less than or equal
# binary_img[condition] = [255, 255, 255]  # Set matching pixels to white
# binary_img[~condition] = [0, 0, 0]  # Set non-matching pixels to black
binary_img[condition2] = [255,255,255]
binary_img[~condition] = [255,255,255]

contours = f_contours(binary_img)
# Count the number of contours (objects)
object_count = len(contours)
print(f"1. The number of objects in the image: {object_count}")

if object_count < 2 or object_count == 1:
    binary_img = copy.deepcopy(masked_image)
    # Apply condition to modify pixel values
    condition = binary_img[:, :, 2] <= int(r_most_frequent_value*0.95)  # channel values less than or equal
    condition2 = binary_img[:, :, 2] == 0 # channel values less than or equal
    # binary_img[condition] = [255, 255, 255]  # Set matching pixels to white
    # binary_img[~condition] = [0, 0, 0]  # Set non-matching pixels to black
    binary_img[condition2] = [255,255,255]
    binary_img[~condition] = [255,255,255]

    contours = f_contours(binary_img)
    object_count = len(contours)
    print(f"2. The number of objects in the image: {object_count}")

# RGB 채널 데이터 추출
r_channel_data = masked_image[:, :, 2]  # OpenCV는 BGR 순서로 채널을 가지므로 R 채널은 2
g_channel_data = masked_image[:, :, 1]  # G 채널은 1
b_channel_data = masked_image[:, :, 0]  # B 채널은 0

# Create subplots using GridSpec
fig = plt.figure(figsize=(15, 6))
grid = plt.GridSpec(2, 3, wspace=0.1, hspace=0.3)

# Plot R channel histogram
ax1 = plt.subplot(grid[0, 0])
ax1.hist(r_channel_data.flatten(), bins=100, range=(1, 100), color='r', alpha=0.7)
ax1.set_title("R Channel Distribution")
ax1.set_xlabel("Pixel Value")
ax1.set_ylabel("Frequency")
ax1.set_xticks(np.arange(1, 101, 10))  # x 축 눈금을 10 단위로 설정

# Plot G channel histogram
ax2 = plt.subplot(grid[0, 1])
ax2.hist(g_channel_data.flatten(), bins=100, range=(1, 100), color='g', alpha=0.7)
ax2.set_title("G Channel Distribution")
ax2.set_xlabel("Pixel Value")
ax2.set_ylabel("Frequency")
ax2.set_xticks(np.arange(1, 101, 10))  # x 축 눈금을 10 단위로 설정

# Plot B channel histogram
ax3 = plt.subplot(grid[0, 2])
ax3.hist(b_channel_data.flatten(), bins=100, range=(1, 100), color='b', alpha=0.7)
ax3.set_title("B Channel Distribution")
ax3.set_xlabel("Pixel Value")
ax3.set_ylabel("Frequency")
ax3.set_xticks(np.arange(1, 101, 10))  # x 축 눈금을 10 단위로 설정

# Plot RGB channel histograms together
ax4 = plt.subplot(grid[1, 0])
ax4.hist(r_channel_data.flatten(), bins=100, range=(1, 100), color='r', alpha=0.5, label='R Channel')
ax4.hist(g_channel_data.flatten(), bins=100, range=(1, 100), color='g', alpha=0.7, label='G Channel')
ax4.hist(b_channel_data.flatten(), bins=100, range=(1, 100), color='b', alpha=0.5, label='B Channel')
ax4.set_title("RGB Channel Distribution")
ax4.set_xlabel("Pixel Value")
ax4.set_ylabel("Frequency")
ax4.set_xticks(np.arange(1, 101, 10))  # x 축 눈금을 10 단위로 설정
ax4.legend()

# Plot binary image
ax5 = plt.subplot(grid[1, 1])
ax5.imshow(image)
ax5.set_title("origin Image")
ax5.axis('off')

# Plot binary image
ax6 = plt.subplot(grid[1, 2])
ax6.imshow(binary_img)
ax6.set_title("binary Image")
ax6.axis('off')
# 결과 이미지 출력
plt.show()

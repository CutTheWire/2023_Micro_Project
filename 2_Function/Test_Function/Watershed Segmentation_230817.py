import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt

# Image loading
img = cv2.imread("./WIN_20230808_09_55_19_Pro.jpg")

alpha = 2.3  # 명암 조정값

# Noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bin_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Image grayscale conversion and contrast adjustment
bin_img = np.clip((1 + alpha) * img - 128 * alpha, 0, 255).astype(np.uint8)

# dilate area
dilate_img = cv2.dilate(bin_img, kernel, iterations=5)

binary_img = cv2.cvtColor(dilate_img, cv2.COLOR_BGR2RGB)
binary_img_copy = copy.deepcopy(binary_img)

# Apply condition to modify pixel values
condition = binary_img[:, :, 0] <= 70  # R channel values less than or equal
binary_img[~condition] = [255, 255, 255]

# Create subplots using GridSpec
fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(2, 3, wspace=0.1, hspace=0.1)

# RGB 채널 데이터 추출
r_channel_data = binary_img_copy[:, :, 0]
g_channel_data = binary_img_copy[:, :, 1]
b_channel_data = binary_img_copy[:, :, 2]

# Plot original image
ax1 = plt.subplot(grid[0, 0])
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("origin Image")
ax1.axis('off')

# Plot cv2.dilate image
ax2 = plt.subplot(grid[0, 1])
ax2.imshow(cv2.cvtColor(dilate_img, cv2.COLOR_BGR2RGB))
ax2.set_title("cv2.dilate Image")
ax2.axis('off')

# Plot binary image
ax3 = plt.subplot(grid[0, 2])
ax3.imshow(binary_img)
ax3.set_title("binary Image")
ax3.axis('off')

# Plot RGB channel histograms together
ax4 = plt.subplot(grid[1, :])
ax4.hist(r_channel_data.flatten(), bins=100, range=(0, 96), color='r', alpha=0.5, label='R Channel')
ax4.hist(g_channel_data.flatten(), bins=100, range=(0, 71), color='g', alpha=0.7, label='G Channel')
ax4.hist(b_channel_data.flatten(), bins=100, range=(0, 96), color='b', alpha=0.5, label='B Channel')
ax4.set_title("R and B Channel Distribution")
ax4.set_xlabel("Pixel Value")
ax4.set_ylabel("Frequency")
ax4.set_xticks(np.arange(0, 96, 5))  # x 축 눈금을 5 단위로 설정
ax4.legend()

plt.tight_layout()
plt.show()

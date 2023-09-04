import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from collections import Counter


# Image loading
img_o = cv2.imread("C:/Users/sjmbe/TW/Micro/230825/3.jpg")
height, width = img_o.shape[:2]

img_o = img_o[int(height*0.01):int(height*0.99):,int(width*0.01):int(width*0.99)]

# 밝기 조절할 값 설정 (음수 값은 어둡게, 양수 값은 밝게 조절)
brightness_factor = -30
alp = 1.0
# Noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_c = copy.deepcopy(img_o)

# 이미지 밝기 조절
adjusted_image = np.int16(img_c) + brightness_factor
adjusted_image = np.clip((1+alp) * adjusted_image - 128 * alp, 0, 255).astype(np.uint8)
adjusted_image = cv2.dilate(adjusted_image, kernel, iterations=1)
cv2.imshow('d',adjusted_image)
# 픽셀 값 0 미만인 경우 0으로 클리핑
img_c = np.clip(adjusted_image, 0, 255).astype(np.uint8)

binary_img = copy.deepcopy(img_c)
# Apply condition to modify pixel values
condition = binary_img [:, :, 1] <= 60  # channel values less than or equal
binary_img [~condition] = [255, 255, 255]

binary_gray = copy.deepcopy(binary_img)

# 색상 반전 수행
binary_gray = cv2.bitwise_not(binary_gray)

# 이진화 수행
threshold_value = 110  # 이진화 임계값 설정
min_contour_area = 50  # 일정 면적 이하는 제외할 윤곽선

_, binary_gray = cv2.threshold(binary_gray, threshold_value, 255, cv2.THRESH_BINARY)
gray_image = cv2.cvtColor(binary_gray, cv2.COLOR_BGR2GRAY)
# 윤곽선 찾기
contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 면적 기준으로 작은 윤곽선 제거
filtered_contours = [contour for contour in contours if min_contour_area < cv2.contourArea(contour)]

contour_areas = []
for contour in filtered_contours:
    contour_area = cv2.contourArea(contour)
    contour_areas.append(contour_area)

# 50의 배수에 따라 그룹으로 나눔
grouped_data = {}
for value in contour_areas:
    group = int(value // 50) * 50
    if group in grouped_data:
        grouped_data[group].append(value)
    else:
        grouped_data[group] = [value]

# 각 그룹의 빈도수 계산
group_frequencies = Counter(grouped_data.keys())

# 빈도수가 가장 높은 두 그룹 선택
top_groups = group_frequencies.most_common(2)

# 선택된 그룹들의 평균 계산
averages = []
for group, _ in top_groups:
    values_in_group = grouped_data[group]
    group_average = np.mean(values_in_group)
    averages.append(group_average)

# 그룹의 데이터를 평탄하게 추출
all_group_values = [value for values in grouped_data.values() for value in values]

# 데이터를 정렬한 후 상위 10%와 하위 10%에 해당하는 값을 제외한 나머지 값들을 선택
sorted_values = sorted(all_group_values)
total_values = len(sorted_values)
top_10_percent = int(total_values * 0.1)
bottom_10_percent = int(total_values * 0.1)
middle_values = sorted_values[top_10_percent:total_values - bottom_10_percent]

# 상위 10%와 하위 10%를 제외한 값들의 평균 계산
filtered_average = np.mean(middle_values)

# 결과 출력
print("Top 2 group averages:", averages)
print("Filtered average:", filtered_average)

# Count the occurrences of each contour area
contour_area_counts = np.bincount(np.array(contour_areas, dtype=int))

# 검출된 윤곽선 그리기
contour_image = copy.deepcopy(img_o)

# 윤곽선 색칠
for contour, contour_area in zip(filtered_contours, contour_areas):
    if contour_area > filtered_average * 1.34:
        color = (255, 0, 0)  # 파란색
    else:
        color = (0, 255, 0)  # 초록색
    cv2.drawContours(contour_image, [contour], -1, color, -1)

green_contours = 0  # 초록색으로 칠해진 윤곽선 개수 초기화
red_areas = []
red_dict = {}

def num(x):
    return (4/3)*x - 1.3


for contour, contour_area in zip(filtered_contours, contour_areas):
    if contour_area > filtered_average * 1.34:
        if contour_area < filtered_average*num(3):
            red_dict[contour_area] = 2
        else:
            for i in range(3,10):
                if filtered_average*num(i) <= contour_area < filtered_average*num(i+1):
                    red_dict[contour_area] = i
print(red_dict)

for contour, contour_area in zip(filtered_contours, contour_areas):
    if contour_area > filtered_average * 1.34:
        green_contours += red_dict[contour_area]  # 초록색 윤곽선 개수 증가
        red_areas.append(contour_area)
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        text = str(green_contours)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_org = (cX - text_size[0] // 2, cY + text_size[1] // 2)
        cv2.putText(contour_image, str(green_contours), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0 ,0), 2)

    else:
        green_contours += 1  # 초록색 윤곽선 개수 증가
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        text = str(green_contours)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_org = (cX - text_size[0] // 2, cY + text_size[1] // 2)
        cv2.putText(contour_image, str(green_contours), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    

print("초록색으로 칠해진 윤곽선 개수:", green_contours)

# Create subplots using GridSpec
fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(2, 2, wspace=0.1, hspace=0.1)

# RGB 채널 데이터 추출
r_channel_data = img_c[:, :, 0]
g_channel_data = img_c[:, :, 1]
b_channel_data = img_c[:, :, 2]



ax4 = plt.subplot(grid[:, :])
ax4.imshow(contour_image)
ax4.set_title("contour Image")
ax4.axis('off')

fig = plt.figure(figsize=(8, 4))
grid = plt.GridSpec(1, 1, wspace=0.1, hspace=0.1)
# Plot normal distribution graph
ax5 = plt.subplot(grid[0, 0])
# Plot RGB channel histograms together
ax5.hist(r_channel_data.flatten(), bins=100, range=(1, 96), color='r', alpha=0.5, label='R Channel')
ax5.hist(g_channel_data.flatten(), bins=100, range=(1, 71), color='g', alpha=0.7, label='G Channel')
ax5.hist(b_channel_data.flatten(), bins=100, range=(1, 96), color='b', alpha=0.5, label='B Channel')
ax5.set_title("R and B Channel Distribution")
ax5.set_xlabel("Pixel Value")
ax5.set_ylabel("Frequency")
ax5.set_xticks(np.arange(1, 96, 5))  # x 축 눈금을 5 단위로 설정
ax5.legend()

# Fit normal distribution parameters
mu, std = norm.fit(contour_areas)
# Create subplots using GridSpec
fig = plt.figure(figsize=(8, 4))
grid = plt.GridSpec(1, 1, wspace=0.1, hspace=0.1)
# Plot normal distribution graph
ax6 = plt.subplot(grid[0, 0])
# Create a range of x values for the curve
x_range = np.linspace(min(contour_areas), max(contour_areas), 100)
# Plot histogram of contour areas
hist, bins, _ = ax6.hist(contour_areas, bins=20, density=False, alpha=0.6, color='c', label='Histogram')

# Find the two most frequent bins (groups)
most_frequent_bins = np.argsort(hist)[-2:]

# Plot normal distribution curve scaled by the total number of contours
norm_curve = norm.pdf(x_range, mu, std) * len(contour_areas) * np.diff(bins)[0]
ax6.plot(x_range, norm_curve, 'r', label='Normal Distribution')
# Add counts on top of the bars
for count, x in zip(hist, bins[:-1]):
    ax6.text(x + np.diff(bins)[0] / 2, count, str(int(count)), ha='center', va='bottom')
ax6.set_title("Contour Area Distribution (Normal Distribution)")
ax6.set_xlabel("Contour Area")
ax6.set_ylabel("Number of Contours")
ax6.legend()



plt.tight_layout()
plt.show()

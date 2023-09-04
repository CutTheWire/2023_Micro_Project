import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from collections import Counter

def contour_areas_avg(filtered_contours):
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

    # 선택된 그룹들의 평균 계산
    averages = []
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
    return averages, filtered_average, contour_areas

# Image loading
img_o = cv2.imread("./1.jpg")
height, width = img_o.shape[:2]

img_o = img_o[int(height*0.01):int(height*0.99):,int(width*0.01):int(width*0.99)]

# 밝기 조절할 값 설정 (음수 값은 어둡게, 양수 값은 밝게 조절)
brightness_factor = -30
alp = 1.0

# Noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
img_c = copy.deepcopy(img_o)

# 이미지 밝기 조절
adjusted_image = np.int16(img_c) + brightness_factor
adjusted_image = np.clip((1+alp) * adjusted_image - 128 * alp, 0, 255).astype(np.uint8)
adjusted_image = cv2.dilate(adjusted_image, kernel, iterations=1)

# 픽셀 값 0 미만인 경우 0으로 클리핑
img_c = np.clip(adjusted_image, 0, 255).astype(np.uint8)
binary_img = copy.deepcopy(img_c)
channel_data = binary_img[:,:,1]

# 0을 제외한 값들의 빈도수 계산
values = channel_data[channel_data != 0]
value_counts = np.bincount(values)

valid_indices = np.where((value_counts >= 200) & (value_counts <= 2500))[0]
sorted_indices = np.argsort(value_counts[valid_indices])  # 빈도수 낮은 순서로 정렬된 값의 인덱스
lowest_indices = valid_indices[sorted_indices][:100]

condition = binary_img [:, :, 1] <= np.average(lowest_indices)*1.05# channel values less than or equal
binary_img [~condition] = [255,255,255]

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
averages, filtered_average, contour_areas = contour_areas_avg(filtered_contours)

# Count the occurrences of each contour area
contour_area_counts = np.bincount(np.array(contour_areas, dtype=int))

# 검출된 윤곽선 그리기
contour_image = copy.deepcopy(img_c)
threshold_average = filtered_average * 1.34
new_contours = ()

g_color = (0, 255, 0)  # 초록색

# 윤곽선 색칠
for contour, contour_area in zip(filtered_contours, contour_areas):
    if contour_area > threshold_average:
        contour_zeros_image = np.zeros_like(img_o)
        cv2.drawContours(contour_zeros_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        contour_zeros_imag  = cv2.dilate(contour_zeros_image, kernel, iterations=3)
        result = cv2.bitwise_and(img_o, contour_zeros_image)
        
        channel_data = result[:,:,1]
        filtered_data = channel_data[(channel_data >= 100) & (channel_data <= 2500)]
        filtered_data_sorted_max = np.sort(filtered_data)[::-1]
        filtered_data_sorted_min = np.sort(filtered_data)
        filter_value = filtered_data_sorted_max[0]*0.67+filtered_data_sorted_min[0]*0.33

        condition = result[:, :, 1] <= filter_value# channel values less than or equal
        result[~condition] = [0,0,0]
        # Convert the modified 'result' array to grayscale
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Perform thresholding to obtain a binary image
        _, result_binary = cv2.threshold(result_gray, 0, 255, cv2.THRESH_BINARY)
        # Find contours in the binary image
        result_contours, _ = cv2.findContours(result_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        try:
            for i in result_contours:
                if cv2.contourArea(i) < 520:
                    cv2.drawContours(contour_image, [i], -1, g_color, -1)
                    new_contours += (i,)
                else:
                    contour_image_c = copy.deepcopy(img_c)
                    mask = np.zeros_like(contour_image)
                    cv2.drawContours(mask, [i], -1, (255, 255, 255), -1)

                    # 기존 이미지에서 해당 윤곽선 부분만 추출
                    masked_region = cv2.bitwise_and(contour_image_c , mask)
                    # 마스크된 영역의 빨간(R) 채널 데이터 추출
                    channel_data = masked_region[:, :, 2]
                    
                    # 특정 범위에 있는 데이터 선택
                    filtered_data = channel_data[(channel_data >= 1) & (channel_data <= 200)]
                    filtered_data_sorted_max = np.sort(filtered_data)[::-1]
                    filtered_data_sorted_min = np.sort(filtered_data)
                    filter_value = filtered_data_sorted_max[0]*0.7+filtered_data_sorted_min[0]*0.3
                    condition2 = masked_region[:, :, 2] <= filter_value
                    masked_region[~condition2] = [0,0,0]
                    
                    # 윤곽선 추출
                    gray_masked_region = cv2.cvtColor(masked_region, cv2.COLOR_BGR2GRAY)
                    contours, _ = cv2.findContours(gray_masked_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # 추출된 윤곽선을 원본 이미지에 그리기
                    for i in contours:
                        if cv2.contourArea(i) < 200:
                            cv2.drawContours(masked_region, [i], -1, g_color, -1)
                            cv2.drawContours(contour_image, [i], -1, g_color, -1)
                            new_contours += (i,)
                            # 결과 이미지 표시
                            # cv2.imshow('1', masked_region)
                            # cv2.waitKey(0)  # 키 입력 대기
                            
                            # # 창 닫기
                            # cv2.destroyAllWindows()

                        else:
                            print("진입")
                            contour_image_c = copy.deepcopy(img_c)
                            mask = np.zeros_like(contour_image)
                            cv2.drawContours(mask, [i], -1, (255, 255, 255), -1)

                            # 기존 이미지에서 해당 윤곽선 부분만 추출
                            masked_region_ = cv2.bitwise_and(contour_image_c , mask)
                            # 마스크된 영역의 빨간(R) 채널 데이터 추출
                            channel_data = masked_region_[:, :, 1]
                            
                            # 특정 범위에 있는 데이터 선택
                            filtered_data = channel_data[(channel_data >= 1) & (channel_data <= 200)]
                            filtered_data_sorted_max = np.sort(filtered_data)[::-1]
                            filtered_data_sorted_min = np.sort(filtered_data)
                            filter_value = filtered_data_sorted_max[0]*0.7+filtered_data_sorted_min[0]*0.3
                            condition2 = masked_region_[:, :, 1] <= filter_value
                            masked_region_[~condition2] = [0,0,0]
                            
                            # 윤곽선 추출
                            gray_masked_region = cv2.cvtColor(masked_region_, cv2.COLOR_BGR2GRAY)
                            contours_, _ = cv2.findContours(gray_masked_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for j in contours_:
                                cv2.drawContours(contour_image, [j], -1, g_color, -1)
                                new_contours += (j,)
                                # 결과 이미지 표시
                            # cv2.imshow('2', masked_region_)
                            # cv2.waitKey(0)  # 키 입력 대기
                            
                            # # 창 닫기
                            # cv2.destroyAllWindows()

        except:
            cv2.drawContours(contour_image, [contour], -1, g_color, -1)
            new_contours += (contour,)

    else:
        cv2.drawContours(contour_image, [contour], -1, g_color, -1)
        new_contours += (contour,)

averages, filtered_average, contour_areas = 0, 0, 0
averages, filtered_average, contour_areas = contour_areas_avg(new_contours)
threshold_average = filtered_average * 1.7
unit_contours = 0  # 초록색으로 칠해진 윤곽선 개수 초기화
red_areas = []
red_dict = {}

for contour, contour_area in zip(new_contours, contour_areas):
    if contour_area > threshold_average:
        if contour_area < 380:
            red_dict[contour_area] = 2
        else:
            if 380 <= contour_area < 520:
                red_dict[contour_area] = 3
            else:
                red_dict[contour_area] = 4
print(red_dict)

for contour, contour_area in zip(new_contours, contour_areas):
    if contour_area > threshold_average:
        unit_contours += red_dict[contour_area]  # 윤곽선 개수 증가
        red_areas.append(contour_area)
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        text = str(unit_contours)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        text_org = (cX - text_size[0] // 2, cY + text_size[1] // 2)
        cv2.putText(contour_image, f"{unit_contours}({red_dict[contour_area]})", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0 ,0), 1)

    elif contour_area > 0:
        unit_contours += 1  # 윤곽선 개수 증가
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        text = str(unit_contours)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_org = (cX - text_size[0] // 2, cY + text_size[1] // 2)
        cv2.putText(contour_image, str(unit_contours), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

print("초록색으로 칠해진 윤곽선 개수:", unit_contours)

# Create subplots using GridSpec
fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(2, 2, wspace=0.1, hspace=0.1)

ax4 = plt.subplot(grid[:, :])
ax4.imshow(contour_image)
ax4.set_title("contour Image")
ax4.axis('off')

plt.tight_layout()
plt.show()

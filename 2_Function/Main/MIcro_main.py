import cv2
import sys
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import copy
import ctypes
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

import image_save as IS

# 초록색 범위 설정 (HSV 순서로 설정)
lower_green = np.array([45, 80, 80])
upper_green = np.array([90, 255, 255])

# 자주색 범위 설정 (HSV 순서로 설정)
lower_purple = np.array([145, 100, 100])
upper_purple = np.array([195, 255, 255])

Line = """
=================================================================================
"""

main_area, sub_area = None, None



def green_line(frame: np.ndarray) -> np.ndarray:
    # BGR을 HSV로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 초록색 범위 내의 영역 찾기
    mask_G = cv2.inRange(hsv, lower_green, upper_green)
    # 초록색 컨투어 찾기
    contours_G, _ = cv2.findContours(mask_G, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 자주색 범위 내의 영역 찾기
    mask_P = cv2.inRange(hsv, lower_purple, upper_purple)
    # 자주색 컨투어 찾기
    contours_P, _ = cv2.findContours(mask_P, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어 그리기 (내부 윤곽선만)
    for contour in contours_G:
        cv2.drawContours(frame, [contour], -1, (125, 205, 55), 16)

    for contour in contours_P:
        cv2.drawContours(frame, [contour], -1, (255, 0, 255), 8)

    return frame

# 윤곽선 사각형으로 다듬는 함수
def approximate_contour(contour: np.ndarray, epsilon_ratio: float = 0.04) -> np.ndarray:
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx

# 사각형 꼭지점 구하는 함수
def find_rect_corners(contour):
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx_pos = cv2.approxPolyDP(contour, epsilon, True)
    return approx_pos

def cv2_to_tk(image):
    # OpenCV 이미지를 PIL 이미지로 변환
    pil_image = Image.fromarray(image)
    # PIL 이미지를 Tkinter 이미지로 변환
    tk_image = ImageTk.PhotoImage(pil_image)
    return tk_image

def resize_image(image, new_width):
    # 현재 이미지의 크기 가져오기
    height, width, _ = image.shape
    # 새로운 크기 계산
    new_height = int((new_width / width) * height)
    # 크기 조정
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def find_nearest_approx(approx_pos_ori, pupple_pos_ori):
    approx_pos_copy = copy.deepcopy(approx_pos_ori)
    pupple_pos_copy = copy.deepcopy(pupple_pos_ori)

    pupple_pos_copy[0][0][1] = pupple_pos_copy[0][0][1]*1.08
    pupple_pos_copy[1][0][1] = pupple_pos_copy[1][0][1] + pupple_pos_copy[0][0][1]*0.06

    # 각 Pupple_pos 좌표에 대해 가장 근사한 좌표 찾기
    for pupple_coord in pupple_pos_copy:
        min_distance = float('inf')
        closest_index = None
        
        for i in range(len(approx_pos_copy)):
            distance = np.linalg.norm(approx_pos_copy[i] - pupple_coord)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        # 가장 근사한 좌표 대체
        approx_pos_copy[closest_index] = pupple_coord
    return approx_pos_copy

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def button_clicked():
    global main_area, sub_area
    main_area, sub_area = None, None

    contour_number = 0
    area_dictionary_G = {}
    area_dictionary_P = {}
    location_nparr = []
    
    _, frame = cap.read()
    approx_frame = copy.deepcopy(frame)

    hsv = cv2.cvtColor(approx_frame, cv2.COLOR_BGR2HSV)

    # 초록색 범위 내의 영역 찾기
    mask_G = cv2.inRange(hsv, lower_green, upper_green)
    mask_P = cv2.inRange(hsv, lower_purple, upper_purple)

    # 컨투어 찾기
    contours_G, _ = cv2.findContours(mask_G, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_P, _ = cv2.findContours(mask_P, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    output_list.insert(tk.END, f"{Line}")

    for contour_G in contours_G:
        area = cv2.contourArea(contour_G)
        if area >= 100:
            contour_number += 1
            area_dictionary_G[area] = contour_G

    for contour_P in contours_P:
        area = cv2.contourArea(contour_P)
        if area >= 100:
            contour_number += 1
            area_dictionary_P[area] = contour_P

    area_dictionary_G = dict(sorted(area_dictionary_G.items(), reverse=True))
    if list(area_dictionary_G.values())[1] is not None:
        main_area = list(area_dictionary_G.values())[1]

    area_dictionary_P = dict(sorted(area_dictionary_P.items(), reverse=True))
    if list(area_dictionary_P.values())[0] is not None:
        sub_area = list(area_dictionary_P.values())[0]

    if main_area is None:
        output_list.insert(tk.END, f"main Area is None!!!")
        output_list.insert(tk.END, f"RETRY!!!")
        output_list.insert(tk.END, "ㅤ")
    else:
        approx_main_area = approximate_contour(main_area)
        approx_sub_area = approximate_contour(sub_area)

        approx_pos_G = find_rect_corners(approx_main_area)
        approx_pos_P = find_rect_corners(approx_sub_area)
        approx_pos_main = find_nearest_approx(approx_pos_G, approx_pos_P)

        if len(approx_pos_main) == 4:  # 초록색이 사각형
            approx_pos_frame = copy.deepcopy(approx_frame)

            for corner in approx_pos_G:
                x, y = corner[0]
                cv2.circle(approx_pos_frame, (x, y), 5, (105, 0, 255), -1)

            for corner in approx_pos_main:
                x, y = corner[0]
                cv2.circle(approx_pos_frame, (x, y), 5, (255, 105, 0), -1)
                location_nparr.append([x, y])

            location = np.array(location_nparr, np.float32)
            location2 = np.array([[0, 1200], [900, 1200], [900, 0], [0, 0]], np.float32)
            pers = cv2.getPerspectiveTransform(location, location2)
            dst = cv2.warpPerspective(approx_frame, pers, (900, 1200))
            dst = cv2.flip(dst, 0)

        approx_pos_frame = resize_image(approx_pos_frame, 1600)
        IS.image_save(dst,"Origin")
        output_list.insert(tk.END, "Origin Image Save")

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
        img_o = dst
        height, width = img_o.shape[:2]

        img_o = img_o[int(height*0.02):int(height*0.99):,int(width*0.02):int(width*0.99)]

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
        threshold_average = filtered_average * 1.35
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
                filter_value = filtered_data_sorted_max[0]*0.68+filtered_data_sorted_min[0]*0.32

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
                        if cv2.contourArea(i) < 400 and 50 < cv2.contourArea(i) :
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
                            filter_value = filtered_data_sorted_max[0]*0.605+filtered_data_sorted_min[0]*0.395
                            condition2 = masked_region[:, :, 2] <= filter_value
                            masked_region[~condition2] = [0,0,0]
                            
                            # 윤곽선 추출
                            gray_masked_region = cv2.cvtColor(masked_region, cv2.COLOR_BGR2GRAY)
                            contours, _ = cv2.findContours(gray_masked_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            # 추출된 윤곽선을 원본 이미지에 그리기
                            for i in contours:
                                if cv2.contourArea(i) < threshold_average and 50 < cv2.contourArea(i):
                                    cv2.drawContours(masked_region, [i], -1, g_color, -1)
                                    cv2.drawContours(contour_image, [i], -1, g_color, -1)
                                    new_contours += (i,)
                                    # 결과 이미지 표시
                                    # cv2.imshow('1', masked_region)
                                    # cv2.waitKey(0)  # 키 입력 대기
                                    
                                    # # 창 닫기
                                    # cv2.destroyAllWindows()

                                else:
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
                                    filter_value = filtered_data_sorted_max[0]*0.5+filtered_data_sorted_min[0]*0.5
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
                if contour_area < threshold_average*2:
                    red_dict[contour_area] = 2
                else:
                    if threshold_average*2 <= contour_area < 480:
                        red_dict[contour_area] = 3
                    else:
                        red_dict[contour_area] = 4

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
            
        # Create subplots using GridSpec
        fig = plt.figure(figsize=(12, 6))
        grid = plt.GridSpec(2, 2, wspace=0.1, hspace=0.1)

        ax4 = plt.subplot(grid[:, :])
        ax4.imshow(contour_image)
        ax4.set_title("contour Image")
        ax4.axis('off')

        plt.text(0.5, -0.1, f"Parts Count : {unit_contours}", transform=ax4.transAxes, ha="center")
        IS.image_save(contour_image,"Scan")
        output_list.insert(tk.END, "Scan Image Save")
        output_list.insert(tk.END, f"Parts Count : {unit_contours} \n")
        plt.tight_layout()
        plt.show()

def exit_clicked():
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

# 카메라 열기
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0은 기본 카메라 장치 번호, 더 많은 카메라가 있는 경우에는 1, 2 등을 시도해볼 수 있습니다.
new_width = 3840
new_height = 2160
cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

root = tk.Tk()
root.title("Micro_TWCV")
root.configure(bg="#666666")
root.state('zoomed')
root.attributes('-fullscreen', True)

# 전체화면 단축키 설정
root.bind("<F11>", lambda event: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

# 대시보드 프레임 생성
dashboard_frame = tk.Frame(root)
dashboard_frame.configure(bg="#666666")
dashboard_frame.pack(fill="both", expand=True)

button_frame = tk.Frame(dashboard_frame)
button_frame.configure(bg="#666666")
button_frame.pack(fill="both", expand=True)  # 좌측에 배치하고 왼쪽 여백 추가

# 버튼 생성 및 스타일 변경
capture_button = tk.Button(button_frame, text="Button", command=button_clicked, height=3, width=10)
capture_button.place(relx=0.5, rely=0.7, anchor="center")  # 상단 여백과 하단 여백 추가
capture_button.configure(bg='#333333', fg="white", font=("Arial", 12, "bold"))

# 버튼 생성 및 스타일 변경
exit_button = tk.Button(button_frame, text="EXIT", command=exit_clicked, height=3, width=8)
exit_button.place(relx=0.97, rely=0.04, anchor="center")  # 상단 여백과 하단 여백 추가
exit_button.configure(bg='#333333', fg="white", font=("Arial", 10, "bold"))

# 비디오 프레임을 표시할 레이블
video_label = tk.Label(dashboard_frame)
video_label.configure(bg="#666666")
video_label.place(relx=0.5, rely=0.35, anchor="center")

output_list = tk.Listbox(button_frame, height=10, width=80)
output_list.place(relx=0.5, rely=0.86, anchor="center")
output_list.configure(bg='#333333', fg="white", font=("Arial", 12, "bold"))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_list.insert(tk.END, f"카메라 해상도: {width}x{height}")

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    image_origin = green_line(frame)
    image_copy = copy.deepcopy(image_origin)

    # 이미지 크기 조정
    image = resize_image(image_copy, 900)
    try:
        # OpenCV 이미지를 Tkinter 이미지로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        video_label.configure(image=image)
        video_label.image = image
        root.update()
    except tk.TclError:
        break

# 카메라 종료
cap.release()
cv2.destroyAllWindows()

# Tkinter 창 실행
root.mainloop()

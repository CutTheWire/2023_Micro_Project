import cv2
import sys
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import copy
import ctypes
from tkinter import PhotoImage
from PIL import Image, ImageTk 

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

def get_monitor_resolution():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


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
    
    # _, frame = cap.read()
    frame = cv2.imread("C:/tinywave_2/2.jpg")
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
            output_list.insert(tk.END, f"{contour_number}. G_Contour Area : {area}\n")
            area_dictionary_G[area] = contour_G

    for contour_P in contours_P:
        area = cv2.contourArea(contour_P)
        if area >= 100:
            contour_number += 1
            output_list.insert(tk.END, f"{contour_number}. P_Contour Area : {area}\n")
            area_dictionary_P[area] = contour_P

    area_dictionary_G = dict(sorted(area_dictionary_G.items(), reverse=True))
    if list(area_dictionary_G.values())[1] is not None:
        main_area = list(area_dictionary_G.values())[1]

    area_dictionary_P = dict(sorted(area_dictionary_P.items(), reverse=True))
    if list(area_dictionary_P.values())[0] is not None:
        sub_area = list(area_dictionary_P.values())[0]

    output_list.insert(tk.END, "ㅤ")

    if main_area is None:
        output_list.insert(tk.END, f"main Area is None!!!")
        output_list.insert(tk.END, f"RETRY!!!")
    else:
        approx_main_area = approximate_contour(main_area)
        approx_sub_area = approximate_contour(sub_area)

        output_list.insert(tk.END, f"#. main Area : {cv2.contourArea(main_area)}\n")
        output_list.insert(tk.END, f"#. approx_main Area : {cv2.contourArea(approx_main_area)}\n")
        
        output_list.insert(tk.END, "ㅤ")

        approx_pos_G = find_rect_corners(approx_main_area)
        # for point in approx_pos_G:
        #     cv2.circle(approx_frame, tuple(point[0]), 25, (125, 0, 55), -1)

        approx_pos_P = find_rect_corners(approx_sub_area)
        # for point in approx_pos_P:
        #     cv2.circle(approx_frame, tuple(point[0]), 25, (255, 0, 255), -1)
        approx_pos_main = find_nearest_approx(approx_pos_G, approx_pos_P)
        
        output_list.insert(tk.END, f"#. approx_pos : {approx_pos_G}\n")
        output_list.insert(tk.END, f"#. Pupple_pos : {approx_pos_P}\n")
        output_list.insert(tk.END, f"#. Main_pos : {approx_pos_main}\n")

        if len(approx_pos_main) == 4:  # 초록색이 사각형
            approx_pos_frame = copy.deepcopy(approx_frame)
            output_list.insert(tk.END, f"#. fix_approx_pos : {approx_pos_main}\n")

            # for corner in approx_pos_G:
            #     x, y = corner[0]
            #     cv2.circle(approx_pos_frame, (x, y), 5, (105, 0, 255), -1)

            for corner in approx_pos_main:
                x, y = corner[0]
                cv2.circle(approx_pos_frame, (x, y), 25, (255, 105, 0), -1)
                location_nparr.append([x, y])

            location = np.array(location_nparr, np.float32)
            location2 = np.array([[0, 1200], [900, 1200], [900, 0], [0, 0]], np.float32)
            pers = cv2.getPerspectiveTransform(location, location2)
            print(type(pers))
            dst = cv2.warpPerspective(approx_frame, pers, (900, 1200))
            dst = cv2.flip(dst, 0)
        # 모니터 해상도를 얻습니다.
        screen_width, screen_height = get_monitor_resolution()

        # 이미지 크기를 얻습니다.
        image_height, image_width, _ = approx_frame.shape

        # 스케일링 팩터를 계산합니다.
        scale_factor = min(screen_width / image_width, screen_height / image_height)

        # 이미지가 화면보다 크다면 크기를 조정합니다.
        if scale_factor < 1:
            approx_frame = cv2.resize(approx_frame, None, fx=scale_factor, fy=scale_factor)

        dst = resize_image(dst, 700)
        approx_pos_frame = resize_image(approx_pos_frame, 1600)

        # 전체화면 모드로 이미지를 보여줍니다.
        cv2.imshow("approx_pos", approx_pos_frame)
        cv2.waitKey(0)
        cv2.imshow("dst", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
root.title("MAIN")
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
capture_button = tk.Button(button_frame, text="Button", command=button_clicked, height=3, width=35)
capture_button.place(relx=0.5, rely=0.68, anchor="center")  # 상단 여백과 하단 여백 추가
capture_button.configure(bg='#333333', fg="white", font=("Arial", 14, "bold"))

# 버튼 생성 및 스타일 변경
exit_button = tk.Button(button_frame, text="EXIT", command=exit_clicked, height=3, width=8)
exit_button.place(relx=0.97, rely=0.04, anchor="center")  # 상단 여백과 하단 여백 추가
exit_button.configure(bg='#333333', fg="white", font=("Arial", 10, "bold"))

# 비디오 프레임을 표시할 레이블
video_label = tk.Label(dashboard_frame)
video_label.configure(bg="#666666")
video_label.place(relx=0.5, rely=0.4, anchor="center")

output_list = tk.Listbox(button_frame, height=10, width=80)
output_list.place(relx=0.5, rely=0.83, anchor="center")
output_list.configure(bg='#333333', fg="white", font=("Arial", 12, "bold"))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_list.insert(tk.END, f"카메라 해상도: {width}x{height}")

while True:
    # 프레임 읽기
    # ret, frame = cap.read()
    # if not ret:
    #     break
    # 이미지를 레이블에 표시하기 위해 PhotoImage 객체 생성
    image = cv2.imread("C:/tinywave_2/2.jpg")

    image_origin = green_line(image)
    image_copy = copy.deepcopy(image_origin)

    # 이미지 크기 조정
    image = resize_image(image_copy, 640)

    # OpenCV 이미지를 Tkinter 이미지로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    try:
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

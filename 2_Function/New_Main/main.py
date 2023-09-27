import sys
import cv2
import numpy as np
import copy
import tkinter as tk
from tkinter import font
from tkinter import messagebox
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Qt5Agg")  # 원하는 백엔드로 변경
import matplotlib.pyplot as plt
from screeninfo import get_monitors

from IMG.camera import Camera
from IMG.Image_Save import save
from IMG.IPP import ImageCV

from Contour.Boundary import BoundaryContour, BoundaryPos, FixPos
from Contour.Unit import Seed

class MainView:
    def __init__(self, root):
        # root style
        self.label_style ={
            'bg': '#333333',
            'fg': 'white',
            'font': font.Font(family="Helvetica", size=18) }

        self.back_style ={
            'bg' : '#666666' }

        # root 설정
        self.root = root
        self.root.title("Micro_TWCV")
        self.root.configure(bg="#666666")
        self.root.state('zoomed')
        self.root.attributes('-fullscreen', True)

        # 전체화면 단축키 설정
        self.root.bind("<F11>", lambda event: self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen")))
        self.root.bind("<Escape>", lambda event: self.root.attributes("-fullscreen", False))

        # GUI 생성
        self.dashboard_frame = tk.Frame(root)
        self.button_frame = tk.Frame(self.dashboard_frame)
        self.video_label = tk.Label(self.dashboard_frame)
        self.capture_button = tk.Button(self.button_frame, text="Check", command=self.Button_click, height=2, width=18)
        self.output_list = tk.Listbox(self.button_frame, height=7, width=80)
        self.exit_button = tk.Button(self.button_frame, text="X", command=self.exit_clicked, height=2, width=4)
        self.text_box = tk.Text(self.button_frame, height=2, width=30)
        self.unit_button = tk.Button(self.button_frame, text="입력",command=self.print_text, height=1, width=3)

        # 디자인
        self.dashboard_frame.configure(self.back_style)
        self.button_frame.configure(self.back_style)
        self.video_label.configure(self.back_style)
        self.capture_button.configure(self.label_style)
        self.output_list.configure(self.label_style)
        self.exit_button.configure(self.label_style)
        self.text_box.configure(self.label_style)
        self.unit_button.configure(self.label_style)
        
        # 위치
        self.dashboard_frame.pack(fill="both", expand=True)
        self.button_frame.pack(fill="both", expand=True)
        self.video_label.place(relx=0.5, rely=0.35, anchor="center")
        self.capture_button.place(relx=0.5, rely=0.7, anchor="center")
        self.output_list.place(relx=0.5, rely=0.86, anchor="center")
        self.exit_button.place(relx=0.97, rely=0.04, anchor="center")  # 상단 여백과 하단 여백 추가
        self.text_box.place(relx=0.5, rely=0.75, anchor="center")
        self.unit_button.place(relx=0.565, rely=0.75, anchor="center")

        self.button_image = None
        self.unit_name = ""
        self.IS = save(self.unit_name, current_button = "Micro", function = "Origin")

    def print_text(self):
        self.unit_name = str(self.text_box.get("1.0", tk.END)).strip()
        if self.unit_name:
            try:
                self.output_list.insert(tk.END, f"제품 : {self.unit_name} 입력 완료")
                self.IS = save(self.unit_name, current_button = "Micro", function = "Origin")
            except:
                self.output_list.insert(tk.END, f"에러 발생 제품 입력을 재시도 해주세요")
        else:
            self.output_list.insert(tk.END, f"제품 입력 안됨, 텍스트 박스에 제품을 입력해 주세요")

    
    def exit_clicked(self):
        result = messagebox.askquestion("Exit Confirmation", "프로그램을 종료 하시겠습니까?")
        if result == "yes":
            cam.release_camera()
            cv2.destroyAllWindows()
            root.destroy()
            sys.exit(0)
            sys.exit(1)
            
    def video_label_update(self, image: np.ndarray):
        self.image = image
        self.video_label.configure(image=image)
        self.video_label.image = image
        self.photo_path_or = ""
        self.photo_path_sc = ""

    def Button_click(self):
        if self.button_image is not None:
            image = self.button_image

            image = IC.Image_Crop(self.button_image, FP.fix_pos, (900,1200))
            image = IC.Image_Slice(image, height_value=0.02, width_value=0.02)

            self.photo_path_or = self.IS.micro_image_save(image)

            self.output_list.insert(tk.END, f"{self.photo_path_or} 경로에 이미지 저장 완료")

            self.Seed_count(image)
            self.output_list.insert(tk.END, f"ㅤ")
        else:
            self.output_list.insert(tk.END, f"이미지 불러오기 실패")
            self.output_list.insert(tk.END, f"ㅤ")

    def Seed_count(self, image: np.ndarray):
        DC_image = copy.deepcopy(image)
        image = IC.Contrast_Adjustment(image)

        SC = Seed(image)
        BGR_list = SC.Find_RGB_Data()
        RGB_data = SC.Filter_RGB_Data(BGR_list[0])
        SC_image = SC.RGB_Mask(RGB_data, index, 1.15)

        IC_image = IC.Histogram_Equalization(SC_image)
        _, contours = SC.Find_Contours(IC_image)
        IC_image = IC.highlight_contours(IC_image, contours)
        IC_image = IC.color_invert(IC_image)
        IC_image = IC.threshold_brightness(IC_image, 40)
        IC_image = IC.Background_Area(IC_image)

        white_parts = IC.White_Mask(DC_image, IC_image)

        # SC = Seed(white_parts)
        # BGR_list = SC.Find_RGB_Data()
        # RGB_data = SC.Filter_RGB_Data(BGR_list[1])
        # white_parts = SC.RGB_Mask(RGB_data, index, 1.)

        white_parts = SC.Black_Contour(white_parts)

        # white_parts = SC.Count_Seed(IC_image, conture)
        IS = save(self.unit_name, current_button = "Micro", function = "Scan") # current_button = "Micro", function = "Micro_Scan"
        self.photo_path_sc = IS.micro_image_save(white_parts)
        self.output_list.insert(tk.END, f"{self.photo_path_sc} 경로에 이미지 저장 완료")

        # result_image_Resolution = IC.Scale_Resolution(white_parts, 0.6)
        # white_parts = cv2.resize(white_parts, result_image_Resolution)

        plt.rcParams['figure.dpi'] = 100
        monitors = get_monitors()
        screen_width, screen_height = monitors[0].width, monitors[0].height
        fig = plt.figure(figsize=(screen_width/100, screen_height/100))

        oringin_image = cv2.imread(self.photo_path_or)
        scan_image = cv2.imread(self.photo_path_sc)

        plt.subplot(1, 2, 1)
        plt.imshow(oringin_image)
        plt.axis('off')  # 선택적으로 축을 표시하지 않도록 설정할 수 있습니다.
        plt.title("Origin")

        # 두 번째 이미지를 오른쪽에 표시
        plt.subplot(1, 2, 2)
        plt.imshow(scan_image)
        plt.axis('off')  # 선택적으로 축을 표시하지 않도록 설정할 수 있습니다.
        plt.title("Scan")
        plt.show()



if __name__ == "__main__":
    cam = Camera()
    cam.open_camera()  # 카메라 열기
    print(cam.cameras_list())
    cam.set_cap_size(2560, 1440)
    root = tk.Tk()
    app = MainView(root)
    app.output_list.insert(tk.END, f"{cam.name} 카메라 활성화")
    app.output_list.insert(tk.END, f"2560x1440 해상도")
    app.output_list.insert(tk.END, f"ㅤ")

    while cam.is_camera_open():
        frame = cam.get_frame()  # 프레임 가져오기
        copy_image = copy.deepcopy(frame)
        app.button_image = copy_image

        # BoundaryScan 클래스 생성
        BC = BoundaryContour(frame)
        IC = ImageCV()
        BP = BoundaryPos()
        try:
            draw_list = []
            contours_list = []

            for index, color in enumerate(["green", "purple"]): # only "green" or "purple" 이외의 색은 구현 안됨
                contours = BC.Find_Contours(color)
                draw_list.append(contours)
                contours_dict = BC.Filter_Contours(contours)

                if index == 0:
                    contours_list.append(BP.Approximate_Pos(list(contours_dict.values())[1]))
                else:
                    equation_contour = list(contours_dict.values())[0]
                    frame, equation_list = BC.FitLine_Contours(equation_contour, frame)

                    contours_list.append(BP.Find_Equation_Pos(equation_list, equation_contour))
            
            # 컨투어를 그려진 이미지 출력
            for index, color in enumerate(["green", "purple"]): # color = "green" or "purple" Other than (55,55,55)
                frame = BC.draw_Contour(draw_list[index], color)

            for index, color in enumerate(["red", "blue"]): # "red", "blue", "green" Other than (55,55,55)
                FP = FixPos(contours_list[index])
                frame = FP.apply_transform(frame, color)
                
            main_pos = BP.Find_Nearest_Pos(contours_list[0], contours_list[1])
            FP = FixPos(main_pos)
        except:
            pass

        # 비디오 레이블 이미지
        fream_Resolution = IC.Scale_Resolution(frame, 0.6)
        video_label_image = cv2.resize(frame, fream_Resolution)

        try:
            image_tk = ImageTk.PhotoImage(
                            Image.fromarray(
                                cv2.cvtColor(video_label_image, cv2.COLOR_BGR2RGB)
                            )) # PIL 이미지를 Tkinter PhotoImage 객체로 변환

            # 레이블에 새 이미지 업데이트
            app.video_label_update(image_tk)
            root.update()

        except tk.TclError:
            pass
        
    # 카메라 종료
    cv2.destroyAllWindows()
    cam.release_camera()
    # Tkinter 창 실행
    root.mainloop()

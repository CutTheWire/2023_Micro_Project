import sys
import cv2
import numpy as np
import re
import copy
import tkinter as tk
from tkinter import font
from tkinter import messagebox
from PyQt5.QtWidgets import QApplication
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

from TW.TWSM import TW
from TW.Loading import LoadingScreen

import bin.logo_data as logo

class MainView:
    def __init__(self, root):
        # root style
        self.label_style ={
            'bg': '#232326',
            'fg': 'white',
            'font': font.Font(family="Helvetica", size=15) }
        
        self.text_label_style ={
            'bg': '#343437',
            'fg': 'white',
            'font': font.Font(family="Helvetica", size=15) }
        
        self.exit_label_style ={
            'bg': '#B43437',
            'fg': 'white',
            'font': font.Font(family="Helvetica", size=15) }

        self.back_style ={
            'bg' : '#343437' }
        
        self.root_style ={
            'bg' : '#565659' }

        # root 설정
        self.root = root
        self.root.title("Micro_TWCV")
        self.root.configure(self.root_style)
        self.root.state('zoomed')
        self.root.attributes('-fullscreen', True)
        
        monitors = get_monitors()
        self.screen_width, self.screen_height = monitors[0].width, monitors[0].height
        self.frame_width = self.screen_width // 3
        self.frame_height = self.screen_height // 20

        self.frame1 = tk.Frame(self.root, width=self.screen_width, height = self.frame_height)
        self.frame2 = tk.Frame(self.root, width=self.frame_width, height = (self.screen_height - self.frame_height))
        self.frame3 = tk.Frame(self.root, width=self.screen_width - self.frame_width, height = (self.screen_height - (self.frame_height*3)))
        self.frame4 = tk.Frame(self.root, width=self.screen_width - self.frame_width, height = self.frame_height)
        self.frame5 = tk.Frame(self.root, width=self.screen_width - self.frame_width, height = self.frame_height)

        # 그리드에 위젯을 배치합니다.
        self.frame1.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="new")
        self.frame2.grid(row=1, column=0, rowspan=3, padx=10, pady=10, sticky="nsew")
        self.frame3.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.frame4.grid(row=2, column=1, padx=10, pady=10, sticky="sew")
        self.frame5.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")

        # 그리드의 크기를 설정합니다.
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_rowconfigure(3, weight=0)
        self.root.grid_columnconfigure(0, weight=10)
        self.root.grid_columnconfigure(1, weight=1)

        # GUI 생성
        self.exit_button = tk.Button(self.frame1, text="X", command=self.exit_clicked, width=4)
        self.output_list = tk.Listbox(self.frame2)
        self.scrollbar = tk.Scrollbar(self.output_list, orient=tk.VERTICAL)
        self.xscrollbar = tk.Scrollbar(self.output_list, orient=tk.HORIZONTAL)
        self.LabelV = tk.Label(self.frame3)
        self.video_label = tk.Label(self.LabelV)
        self.capture_text = tk.Label(self.frame4, text="ㅤㅤ이미지 분석ㅤ")
        self.capture_button = tk.Button(self.frame4, text="Check", command=self.Button_click, width=15)
        self.text_box = tk.Text(self.frame4, height=1,width=25)
        self.unit_button = tk.Button(self.frame4, text="입력",command=self.print_text, width=15)
        self.unit_text = tk.Label(self.frame4, text="ㅤ|ㅤ부품 이름ㅤ")
        self.TW_logo_image = tk.Label(self.frame5)

        # Listbox와 Scrollbar 연결
        self.output_list.config(yscrollcommand = self.scrollbar.set)
        
        self.scrollbar.config(command = self.output_list.yview)
        self.output_list.config(xscrollcommand=self.xscrollbar.set)
        self.xscrollbar.config(command = self.output_list.xview)

        # 엔터 키 이벤트에 대한 바인딩을 설정
        self.text_box.bind("<Return>", lambda e: "break")
        
        image = tk.PhotoImage(data=logo.image_base64)
        # 디자인
        self.frame1.configure(self.back_style)
        self.frame2.configure(self.back_style)
        self.frame3.configure(self.back_style)
        self.frame4.configure(self.back_style)
        self.frame5.configure(self.root_style)

        self.exit_button.configure(self.exit_label_style)
        self.output_list.configure(self.label_style)
        self.video_label.configure(self.back_style)
        self.capture_text.configure(self.text_label_style)
        self.capture_button.configure(self.label_style)
        self.text_box.configure(self.label_style)
        self.unit_button.configure(self.label_style)
        self.unit_text.configure(self.text_label_style)
        self.TW_logo_image.configure(self.root_style, image=image)
        self.TW_logo_image.image = image

        # 위치
        self.exit_button.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_list.pack(fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.LabelV.pack(fill=tk.BOTH, expand=True)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.capture_text.pack(side=tk.LEFT, fill=tk.Y)
        self.capture_button.pack(side=tk.LEFT, fill=tk.Y)
        self.unit_button.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_box.pack(side=tk.RIGHT, fill=tk.Y)
        self.unit_text.pack(side=tk.RIGHT, fill=tk.Y)
        self.TW_logo_image.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.button_image = None
        self.end = None
        self.unit_name = "empty"
        self.IS = save(self.unit_name, program = "Micro", function = "Origin", current_button="")

    def is_valid_input(self, text):
        if re.match(r'^[A-Za-z0-9!@#$%^_.-]*$', text):
            return True
        else:
            return False

    def print_text(self):
        input_text = str(self.text_box.get("1.0", tk.END)).strip()
        if input_text:
            if self.is_valid_input(input_text):
                try:
                    self.unit_name = str(self.text_box.get("1.0", tk.END)).strip()
                    self.output_list.insert(tk.END, f"제품 : {self.unit_name} 입력 완료ㅤㅤ")
                    self.IS = save(self.unit_name, program= "Micro", function = "Origin", current_button= "")
                except:
                    self.output_list.insert(tk.END, f"에러 발생: 제품 입력을 재시도 해주세요ㅤㅤ")
            else:
                self.output_list.insert(tk.END, "ㅤㅤ")
                self.output_list.insert(tk.END, "에러 발생: 제품 입력에 지원하지 않는 문자가 포함되어 있습니다.ㅤㅤ")
                self.output_list.insert(tk.END, "영어 또는 숫자 일부 기호만 사용.ㅤㅤ")
                self.output_list.insert(tk.END, "ㅤㅤ")
                self.output_list.insert(tk.END, "ㅤ! @ # $ % ^ _ - . 만을 사용하여 제품 이름을 입력해 주십시오.ㅤㅤ")
        else:
            self.output_list.insert(tk.END, f"제품 입력 안됨, 텍스트 박스에 제품을 입력해 주세요ㅤㅤ")

    def exit_clicked(self):
        result = messagebox.askquestion("Exit Confirmation", "프로그램을 종료 하시겠습니까?")
        if result == "yes":
            self.end = 0
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

            self.output_list.insert(tk.END, f"{self.photo_path_or} 경로에 이미지 저장 완료ㅤㅤ")

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
        white_parts = SC.Black_Contour(white_parts)

        IS = save(self.unit_name, program= "Micro", function = "Scan", current_button="") # current_button = "Micro", function = "Micro_Scan"
        self.photo_path_sc = IS.micro_image_save(white_parts)
        self.output_list.insert(tk.END, f"{self.photo_path_sc} 경로에 이미지 저장 완료ㅤㅤ")

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
    T = TW()
    if T() == True:
        subapp = QApplication(sys.argv)
        loading_screen = LoadingScreen()
        loading_screen.show()

        root = tk.Tk()
        app = MainView(root)
        cam = Camera()
        cam.open_camera()  # 카메라 열기
        cam.set_cap_size(app.screen_width, (app.screen_width*9)//16)
        app.output_list.insert(tk.END, f"연결된 카메라 리스트")
        app.output_list.insert(tk.END, f"{cam.cameras_list()}ㅤㅤ")
        app.output_list.insert(tk.END, f"ㅤ")
        app.output_list.insert(tk.END, f"{cam.name} 카메라 활성화")
        app.output_list.insert(tk.END, f"카메라 해상도 {app.screen_width}, {(app.screen_width*9)//16}")
        app.output_list.insert(tk.END, f"ㅤ")
        loading_screen.close()

        IC = ImageCV()
        BP = BoundaryPos()

        while cam.is_camera_open():
            frame = cam.get_frame()  # 프레임 가져오기
            copy_image = copy.deepcopy(frame)
            app.button_image = copy_image

            # BoundaryScan 클래스 생성
            BC = BoundaryContour(frame)
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
            except IndexError:
                pass

            except cv2.error:
                if app.end == 0:
                    break
                else:
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
        root.mainloop()
        sys.exit(subapp.exec_())

    elif T() == False:
        messagebox.showinfo("SM ERROR", "해당 프로그램은 설정된 컴퓨터에서 실행 가능합니다.\n변경을 원할 경우 업체에 요청하시길 바랍니다.")

    elif T() == 2:
        messagebox.showinfo("OS ERROR", "해당 프로그램은 Windows10 이상에서만 실행 가능합니다.")

    else:
        messagebox.showinfo("ERROR", T())
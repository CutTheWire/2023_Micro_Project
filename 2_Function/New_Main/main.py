import cv2
import numpy as np
import copy
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

from IMG.camera import Camera
from IMG.Image_Save import save
from IMG.IPP import ImageCV

from Contour.Boundary import BoundaryContour, BoundaryPos, FixPos
from Contour.Unit import Seed

class MainView:
    def __init__(self, root):
        # root style
        label_style ={
            'bg': '#333333',
            'fg': 'white',
            'font': ('Arial', 12, 'bold') }

        back_style ={
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
        self.capture_button = tk.Button(self.button_frame, text="Check", command=self.Button_click, height=3, width=18)
        self.output_list = tk.Listbox(self.button_frame, height=10, width=80)

        # 디자인
        self.dashboard_frame.configure(**back_style)
        self.button_frame.configure(**back_style)
        self.video_label.configure(**back_style)
        self.capture_button.configure(**label_style)
        self.output_list.configure(**label_style)
        
        # 위치
        self.dashboard_frame.pack(fill="both", expand=True)
        self.button_frame.pack(fill="both", expand=True)
        self.video_label.place(relx=0.5, rely=0.35, anchor="center")
        self.capture_button.place(relx=0.5, rely=0.7, anchor="center")
        self.output_list.place(relx=0.5, rely=0.86, anchor="center")

        #---------------------------------------------------------------------------------------------------------------------

        # def Button_click
        self.button_image = None
        self.IS = save(current_button = "TEST", function = "Micro") # current_button = "Micro", function = "Origin"




    def video_label_update(self, image: np.ndarray):
        self.image = image
        self.video_label.configure(image=image)
        self.video_label.image = image

    def Button_click(self):
        for i in range(2):
            self.output_list.insert(tk.END, f"{i+1} : {contours_list[i]}")
        self.output_list.insert(tk.END, f"1차 방정식: y  = {equation_list[0][0]}x + {equation_list[1][0]}")
        self.output_list.insert(tk.END, f"ㅤ")

        if self.button_image is not None:
            image = self.button_image
            image = IC.Image_Crop(self.button_image, FP.fix_pos, (900,1200))
            image = IC.Image_Slice(image, height_value=0.02, width_value=0.02)

            photo_path = self.IS.micro_image_save(image)

            self.output_list.insert(tk.END, f"{photo_path} 경로에 이미지 저장 완료")
            self.output_list.insert(tk.END, f"ㅤ")

            self.Seed_count(image)
            self.output_list.insert(tk.END, f"ㅤ")
        else:
            self.output_list.insert(tk.END, f"이미지 불러오기 실패")
            self.output_list.insert(tk.END, f"ㅤ")

    def Seed_count(self, image: np.ndarray):
        SC = Seed(image)
        BGR_list = SC.Find_RGB_Data()
        RGB_data = SC.Filter_RGB_Data(BGR_list[1])
        image = SC.RGB_Mask(RGB_data, index)
        image = SC.Background_Area(image)
        image, conture = SC.Find_Contours(image)
        image = SC.Count_Seed(image, conture)

        result_image_Resolution = IC.Scale_Resolution(image, 0.5)
        image = cv2.resize(image, result_image_Resolution)
        cv2.imshow("",image)
        cv2.waitKey(0)
    



if __name__ == "__main__":
    cam = Camera()
    cam.open_camera()  # 카메라 열기
    cam.set_cap_size(2560, 1440)

    root = tk.Tk()
    app = MainView(root)

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
        fream_Resolution = IC.Scale_Resolution(frame, 0.3)
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

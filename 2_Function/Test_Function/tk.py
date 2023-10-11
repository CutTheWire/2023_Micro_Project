import sys
import tkinter as tk
from tkinter import ttk
from tkinter import font
from tkinter import messagebox

from screeninfo import get_monitors
import logo_data as logo

class mainTK:
    def __init__(self, root):
        self.label_style ={
            'bg': '#232326',
            'fg': 'white',
            'font': font.Font(family="Helvetica", size=15) }
        
        self.text_label_style ={
            'bg': '#343437',
            'fg': 'white',
            'font': font.Font(family="Helvetica", size=15) }

        self.back_style ={
            'bg' : '#343437' }
        
        self.root_style ={
            'bg' : '#565659' }
        
        self.threshold = tk.DoubleVar() # 슬라이더 값을 저장하는 변수
        self.threshold.set(46.4)
        
        self.threshold_values = [46.4,47.7,59.0,18.1]
        self.threshold_values.sort()  # 임계값 리스트

        # root 설정
        self.root = root
        self.root.title("Micro_TWCV")
        self.root.configure(self.root_style)
        self.root.state('zoomed')
        self.root.attributes('-fullscreen', True)
        
        monitors = get_monitors()
        self.screen_width, self.screen_height = monitors[0].width, monitors[0].height
        self.frame_width = self.screen_width // 10
        self.frame_height = self.screen_height // 20

        self.frame1 = tk.Frame(self.root, width=self.screen_width, height = self.frame_height)
        self.frame2 = tk.Frame(self.root, width=self.screen_width, height = (self.screen_height - self.frame_height*4))
        self.frame3 = tk.Frame(self.root, width=self.screen_width, height = self.frame_height)
        self.frame4 = tk.Frame(self.root, width=self.screen_width, height = self.frame_height)
        self.frame5 = tk.Frame(self.root, width=self.screen_width, height = self.frame_height)

        # 그리드에 위젯을 배치합니다.
        self.frame1.grid(row=0, column=0, padx=10, pady=10, sticky="new")
        self.frame2.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.frame3.grid(row=2, column=0, padx=10, pady=10, sticky="sew")
        self.frame4.grid(row=3, column=0, padx=10, pady=10, sticky="sew")
        self.frame5.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

        # 그리드의 크기를 설정합니다.
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_rowconfigure(3, weight=0)
        self.root.grid_columnconfigure(0, weight=10)

        # GUI 생성
        self.exit_button = tk.Button(self.frame1, text="X", command=lambda: self.exit_clicked(), width=5)
        self.LabelV = tk.Label(self.frame2)
        self.video_label = tk.Label(self.LabelV)

        self.tray_text = tk.Label(self.frame3, text="ㅤㅤ트레이 버튼ㅤㅤ")
        self.ten_by_ten_button = tk.Button(self.frame3, text="10x10", command=lambda: self.button_clicked("10x10"), height=2, width=20)
        self.five_by_ten_button = tk.Button(self.frame3, text="5x10", command=lambda: self.button_clicked("5x10"), height=2, width=20)
        self.S_ten_by_ten_button = tk.Button(self.frame3, text="S_10x10", command=lambda: self.button_clicked("S_10x10"), height=2, width=20)
        self.two_hundred_button = tk.Button(self.frame3, text="200-1", command=lambda: self.button_clicked("200-1"), height=2, width=20)
        self.combobox = ttk.Combobox(self.frame3, values=self.threshold_values, style='TCombobox', width=20)
        self.label_empty = tk.Label(self.frame3, text="ㅤㅤㅤㅤ")
        self.threshold_slider = tk.Scale(self.frame3, from_=0.1, to=300, resolution=0.1, orient=tk.HORIZONTAL, length=600, variable=self.threshold, width=35)
        self.threshold_text = tk.Label(self.frame3, text="ㅤㅤ임계값ㅤㅤ")
        
        self.capture_text = tk.Label(self.frame4, text="ㅤㅤ이미지 분석ㅤㅤ")
        self.capture_button = tk.Button(self.frame4, text="Check", height=2, width=20)
        self.text_box = tk.Text(self.frame4, height=1,width=25)
        self.unit_button = tk.Button(self.frame4, text="입력", height=2, width=20)
        self.unit_text = tk.Label(self.frame4, text="ㅤㅤ부품 이름(한글제외)ㅤㅤ")
        self.TW_logo_image = tk.Label(self.frame5)

        # 엔터 키 이벤트에 대한 바인딩을 설정
        self.text_box.bind("<Return>", lambda e: "break")
        self.combobox.bind("<<ComboboxSelected>>", self.on_combobox_select)
        
        image = tk.PhotoImage(data=logo.image_base64)

        # 디자인
        self.style = ttk.Style()
        self.style.map('TCombobox', fieldbackground=[('readonly', '#232326')])
        self.combobox.configure(style='TCombobox')
        self.combobox['font'] = font.Font(family="Helvetica", size=15)
        self.combobox['foreground'] = 'white'

        self.frame1.configure(self.back_style)
        self.frame2.configure(self.back_style)
        self.frame3.configure(self.back_style)
        self.frame4.configure(self.back_style)
        self.frame5.configure(self.root_style)

        self.exit_button.configure(self.label_style)
        self.video_label.configure(self.back_style)

        self.tray_text.configure(self.text_label_style)
        self.ten_by_ten_button.configure(self.label_style)
        self.five_by_ten_button.configure(self.label_style)
        self.S_ten_by_ten_button.configure(self.label_style)
        self.two_hundred_button.configure(self.label_style)
        self.threshold_text.configure(self.text_label_style)
        self.label_empty.configure(self.text_label_style)
        self.threshold_slider.configure(self.label_style)

        self.capture_text.configure(self.text_label_style)
        self.capture_button.configure(self.label_style)
        self.text_box.configure(self.label_style)
        self.unit_button.configure(self.label_style)
        self.unit_text.configure(self.text_label_style)
        self.TW_logo_image.configure(self.root_style, image=image)
        self.TW_logo_image.image = image
        
        # 위치
        self.exit_button.pack(side=tk.RIGHT, fill=tk.Y)
        self.LabelV.pack(fill=tk.BOTH, expand=True)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.tray_text.pack(side=tk.LEFT, fill=tk.Y)
        self.ten_by_ten_button.pack(side=tk.LEFT, fill=tk.Y)
        self.five_by_ten_button.pack(side=tk.LEFT, fill=tk.Y)
        self.S_ten_by_ten_button.pack(side=tk.LEFT, fill=tk.Y)
        self.two_hundred_button.pack(side=tk.LEFT, fill=tk.Y)
        self.combobox.pack(side=tk.RIGHT, fill=tk.Y)
        self.label_empty.pack(side=tk.RIGHT, fill=tk.Y)
        self.threshold_slider.pack(side=tk.RIGHT, fill=tk.Y)
        self.threshold_text.pack(side=tk.RIGHT, fill=tk.Y)

        self.capture_text.pack(side=tk.LEFT, fill=tk.Y)
        self.capture_button.pack(side=tk.LEFT, fill=tk.Y)
        self.unit_button.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_box.pack(side=tk.RIGHT, fill=tk.Y)
        self.unit_text.pack(side=tk.RIGHT, fill=tk.Y)
        self.TW_logo_image.pack(side=tk.RIGHT, fill=tk.Y)

    def button_clicked(self, button):
        return button
    
    # 콤보박스 선택 시 실행되는 함수
    def on_combobox_select(self, event):
        selected_value = self.combobox.get()
        self.threshold.set(float(selected_value))
        tk.DoubleVar().set(float(selected_value))
    
    def exit_clicked(self):
        result = messagebox.askquestion("Exit Confirmation", "프로그램을 종료 하시겠습니까?")
        if result == "yes":
            self.end = 0
            root.destroy()
            sys.exit(0)

    def on_slider_focus(self, event):
        self.threshold_slider.focus_set(51)

    def on_key_press(self, event):
        if event.keysym == 'Left':
            self.threshold_slider.set(self.threshold_slider.get() - 0.1)
        elif event.keysym == 'Right':
            self.threshold_slider.set(self.threshold_slider.get() + 0.1)


        
if __name__ == "__main__":
    root = tk.Tk()
    mainTK(root)
    root.mainloop()

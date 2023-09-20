import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Tkinter 윈도우 생성
root = tk.Tk()

# OpenCV로 이미지 읽기
image = cv2.imread("C:/tinywave_2/RGB_2.png")

# OpenCV 이미지를 PIL 이미지로 변환
image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# PIL 이미지를 Tkinter PhotoImage 객체로 변환
image_tk = ImageTk.PhotoImage(image_pil)

print(type(image_tk))
# Tkinter 윈도우에 이미지 표시
label = tk.Label(root, image=image_tk)
label.pack()

# Tkinter 메인 루프 시작
root.mainloop()

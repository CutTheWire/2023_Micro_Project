import cv2
import socket
import tkinter as tk
import sys
from tkinter import messagebox
from PIL import Image, ImageTk

import detect as DT
import tkinter_image as TI

host_name = "Saeon_Note"

if socket.gethostname() != host_name:
    messagebox.showwarning("경고", "허용되지 않은 PC입니다.")
    sys.exit()
area_threshold = 500
threshold_value = 100

cap = cv2.VideoCapture(2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)

ret, frame = cap.read()


window = tk.Tk()
window.title("Real-time Object Detection")
window.attributes("-fullscreen", True)

label = tk.Label(window)
label.pack()

DT.detect_objects()

area_threshold_slider = tk.Scale(window, from_=0, to=2000, resolution=10, orient=tk.HORIZONTAL, label="면적 (Area)",
                                length=400)
area_threshold_slider.set(area_threshold)
area_threshold_slider.pack()

threshold_value_slider = tk.Scale(window, from_=0, to=255, resolution=1, orient=tk.HORIZONTAL,label="임계값 (Threshold)",
                                length=400)

threshold_value_slider.set(threshold_value)
threshold_value_slider.pack()

def update_threshold_value(value):
    global threshold_value
    threshold_value = int(value)

threshold_value_slider.config(command=update_threshold_value)

window.mainloop()

cap.release()
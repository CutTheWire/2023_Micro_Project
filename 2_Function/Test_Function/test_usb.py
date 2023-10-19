import tkinter as tk
from pygrabber.dshow_graph import FilterGraph
import cv2
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("카메라 앱")
        self.state = True

        # 윈도우 크기 설정
        self.root.geometry("800x600")

        # 카메라 리스트를 가져오기
        self.cameras = FilterGraph().get_input_devices()

        # 리스트박스 생성
        self.camera_listbox = tk.Listbox(self.root)
        self.camera_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.camera_listbox.bind('<<ListboxSelect>>', self.show_selected_camera)

        # 카메라 목록 업데이트
        self.update_camera_list()

        # OpenCV 비디오 캡처 초기화
        self.cap = 0
        self.video_label = tk.Label(self.root)
        self.video_label.pack(side=tk.RIGHT, padx=10, pady=10)
        self.selected_camera_index = 0

        # 새로고침 버튼 생성
        self.refresh_button = tk.Button(self.root, text="새로고침", command=self.update_camera_list)
        self.refresh_button.pack()

        # 시작 시 첫 번째 카메라 선택
        self.show_selected_camera(0)

    # 카메라 목록 업데이트
    def update_camera_list(self):
        self.camera_listbox.delete(0, tk.END)
        self.cameras = FilterGraph().get_input_devices()
        for index, device_name in enumerate(self.cameras):
            self.camera_listbox.insert(tk.END, f"카메라 {index}: {device_name}")

    # 선택한 카메라로 비디오 스트림 시작
    def show_selected_camera(self, event):
        selected_index = self.camera_listbox.curselection()
        if selected_index:
            self.selected_camera_index = int(selected_index[0])
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.selected_camera_index)
            self.show_video()

    # 비디오 프레임을 보여주기
    def show_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.video_label.configure(image=photo)
            self.video_label.image = photo
        self.root.after(10, self.show_video)  # 10ms 마다 업데이트

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()

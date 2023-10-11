import tkinter as tk
from tkinter import messagebox
import requests

class License_tk:
    def __init__(self) -> None:
        self.window = tk.Tk()
        self.window.title("라이센스 키 확인")
        self.window.attributes("-fullscreen", False)
        self.window.resizable(False, False)

        self.center_frame = tk.Frame(self.window)
        self.company_label = tk.Label(self.center_frame, text="업체를 입력하세요:")
        self.entry_company = tk.Entry(self.center_frame)
        self.key_label = tk.Label(self.center_frame, text="라이센스 키를 입력하세요:")
        self.entry_license_key = tk.Entry(self.center_frame)
        self.button = tk.Button(self.center_frame, text="확인", command=self.check_license_key)

        self.center_frame.pack(expand=True)
        self.company_label.pack(padx=10, pady=10)
        self.entry_company.pack(padx=10, pady=10)
        self.key_label.pack(padx=10, pady=10)
        self.entry_license_key.pack(padx=10, pady=10)
        self.button.pack(padx=10, pady=10)

        # 창 크기 설정 및 창을 화면 중앙에 위치시키기
        self.window_width = 500
        self.window_height = 300

    def screen(self):
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x_position = (screen_width - self.window_width) // 2
        y_position = (screen_height - self.window_height) // 2
        self.window.geometry("{}x{}+{}+{}".format(self.window_width, self.window_height, x_position, y_position))

    # 라이센스 키 확인 함수
    def check_license_key(self):
        key_input = self.entry_license_key.get().strip()
        user_input = self.entry_company.get().strip()
        
        # Replace with the actual URL of your license verification server
        server_url = "http://127.0.0.1:5000/verify_license"  # Use the correct endpoint
        
        # Send a POST request to the server to verify the license key
        response = requests.post(server_url, json={"user": user_input.lower(), "key": key_input})  # Send user and key as JSON
        
        try:
            # Send a POST request to the server to verify the license key
            response = requests.post(server_url, json={"user": user_input.lower(), "key": key_input})  # Send user and key as JSON
            
            if response.status_code == 200:
                # The server responded with a success status code (e.g., 200 OK)
                result = response.json()  # Assuming the server responds with JSON
                if result.get("valid"):
                    messagebox.showinfo("확인", "라이센스 키가 유효합니다.")
                else:
                    messagebox.showerror("오류", "라이센스 키가 유효하지 않습니다.")
            else:
                # The server returned an error status code
                messagebox.showerror("오류", "라이센스 키 확인 중 오류가 발생했습니다.")
        except ConnectionError:
            # Handle the connection error and display an error message
            messagebox.showerror("연결 오류", "서버와의 연결을 확인할 수 없습니다. 서버를 실행 중인지 확인하세요.")


if __name__ == "__main__":
    Ltk = License_tk()
    # Tkinter 창 실행
    Ltk.screen()
    Ltk.window.mainloop()

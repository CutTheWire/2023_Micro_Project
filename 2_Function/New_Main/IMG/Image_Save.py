import os
import cv2
import numpy as np
from datetime import datetime

class save:
    def __init__(self, unit_name: str, program: str, function: str, current_button: str) -> None:
        self.program = program
        self.function = function
        self.unit = unit_name
        self.current_button = current_button
        # Documents(문서) 폴더 경로
        self.documents_folder = os.path.join(os.path.expanduser("~"))
        self.main_folder = os.path.join(self.documents_folder, "TW")
        self.sub_folder = os.path.join(self.main_folder, self.program)
        if self.program == "Tray":
            self.button_folder = os.path.join(self.sub_folder, self.current_button)
            self.date_folder = os.path.join(self.button_folder, datetime.today().strftime('%y%m%d'))
        else:
            self.date_folder = os.path.join(self.sub_folder, datetime.today().strftime('%y%m%d'))
        self.unit_folder = os.path.join(self.date_folder, self.unit)
    
    def micro_image_save(self, frame: np.ndarray) -> str:
        # 폴더가 이미 존재하는지 확인 후 생성
        for f in [self.main_folder, self.sub_folder, self.date_folder, self.unit_folder]:
            if not os.path.exists(f):
                os.mkdir(f)
        
        # 이미지를 읽어옴
        filename = f"{datetime.today().strftime('%H%M%S')}_{self.function}.jpg"
        photo_path = os.path.join(self.unit_folder, filename)
        
        # 이미지를 저장
        cv2.imwrite(photo_path, frame)
        return photo_path
    
    def tray_image_save(self, frame: np.ndarray) -> str:
        # 폴더가 이미 존재하는지 확인 후 생성
        for f in [self.main_folder, self.sub_folder, self.button_folder, self.date_folder, self.unit_folder]:
            if not os.path.exists(f):
                os.mkdir(f)
        
        # 이미지를 읽어옴
        filename = f"{datetime.today().strftime('%H%M%S')}_{self.function}.jpg"
        photo_path = os.path.join(self.unit_folder, filename)
        
        # 이미지를 저장
        cv2.imwrite(photo_path, frame)
        return photo_path

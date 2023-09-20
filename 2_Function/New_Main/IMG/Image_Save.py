import os
import cv2
import numpy as np
from datetime import datetime

class save:
    def __init__(self, current_button: str, function: str) -> None:
        self.current_button = current_button
        self.function = function

        # Documents(문서) 폴더 경로
        self.documents_folder = os.path.join(os.path.expanduser("~"))
        self.main_folder = os.path.join(self.documents_folder, "TW")
        self.sub_folder = os.path.join(self.main_folder, self.current_button)
        self.date_folder = os.path.join(self.sub_folder, datetime.today().strftime('%y%m%d'))

    def get_unique_filename(self, extension: str) -> str:
        # 중복되지 않는 파일 이름을 생성하기 위한 함수
        filename = f"{datetime.today().strftime('%H%M%S')}_{self.function}{extension}"
        return filename
    
    def micro_image_save(self, frame: np.ndarray) -> str:
        # 폴더가 이미 존재하는지 확인 후 생성
        for f in [self.main_folder, self.sub_folder, self.date_folder]:
            if not os.path.exists(f):
                os.mkdir(f)
        
        # 이미지를 읽어옴
        filename = self.get_unique_filename(extension=".jpg")
        photo_path = os.path.join(self.date_folder, filename)
        
        # 이미지를 저장
        cv2.imwrite(photo_path, frame)
        return photo_path

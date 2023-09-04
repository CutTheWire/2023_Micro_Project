import os
import cv2
from datetime import datetime
import numpy as np

current_button = "Micro"

def image_save(frame,function):
    global current_button


    # Documents(문서) 폴더 경로
    documents_folder = os.path.join(os.path.expanduser("~"))
    main_folder = os.path.join(documents_folder, "TW")
    sub_folder = os.path.join(main_folder, current_button)
    date_folder = os.path.join(sub_folder, datetime.today().strftime('%y%m%d'))

    # 폴더가 이미 존재하는지 확인 후 생성
    for f in [main_folder, sub_folder, date_folder]:
        if not os.path.exists(f):
            os.mkdir(f)

    def get_unique_filename(extension: str) -> str:
        # 중복되지 않는 파일 이름을 생성하기 위한 함수
        filename = f"{datetime.today().strftime('%H%M%S')}_{function}{extension}"
        return filename

    # 이미지를 읽어옴
    filename = get_unique_filename(extension = ".jpg")
    photo_path = os.path.join(date_folder, filename)
        # 이미지를 저장
    cv2.imwrite(photo_path, frame)
    print(f"{photo_path}에 사진을 저장했습니다.")
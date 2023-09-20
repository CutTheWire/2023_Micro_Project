import cv2
import numpy as np
import copy

from matplotlib import pyplot as plt

from IMG.IPP import ImageCV

class Seed:
    def __init__(self, frame) -> None:
        self.IC = ImageCV()
        self.image = frame
        
        # def Find_Contours
        self.threshold_area = 100

        # def Find_RGB_Data
        self.R_channel_data = None
        self.G_channel_data = None
        self.B_channel_data = None

        # def Filter_RGB_Data
        self.min_count = 100
        self.max_count = 5000
        self.rank = 500

    def Count_Seed(self, image: np.ndarray, contours: np.ndarray) -> np.ndarray:
        # 검출된 윤곽선의 개수 확인
        num_contours = len(contours)

        # 이미지에 텍스트 추가
        text = f"Number of Contours: {num_contours}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1  # 텍스트 크기를 키움
        font_color = (255, 0, 255)  # 흰색
        thickness = 3  # 텍스트 굵기를 늘림

        # 텍스트를 이미지의 왼쪽 상단 모서리에 그리기
        org = (15, 30)  # 왼쪽 상단 모서리 위치

        # 이미지에 텍스트 추가
        image_with_text = image.copy()
        cv2.putText(image_with_text, text, org, font, font_scale, font_color, thickness)

        return image_with_text

    def Background_Area(self, image: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Distance transform
        dist = cv2.distanceTransform(image, cv2.DIST_L2, 5)
        # foreground area
        _, sure_fg = cv2.threshold(dist, 0.55 * dist.max(), 255, cv2.THRESH_BINARY)
        sure_bg = cv2.dilate(sure_fg, kernel, iterations=1.5)
        sure_bg = sure_bg.astype(np.uint8)
        return sure_bg
    
    def Find_Contours(self, image: np.ndarray) -> np.ndarray:
        # 윤곽선 검출
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # 윤곽선 면적을 계산하고 일정 크기 이하의 윤곽선을 필터링
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.threshold_area]
        contour_image = cv2.drawContours(self.image.copy(), filtered_contours, -1, (255, 0, 255), 2)
        return contour_image, filtered_contours
    
    def Mask(self, image: np.ndarray, contours: np.ndarray) -> np.ndarray:
        # 새로운 마스크 이미지 생성
        mask = np.zeros_like(image)
        # 필터링된 윤곽선을 마스크에 그립니다.
        cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        # 원본 이미지와 마스크를 이용하여 윤곽선 부분만 추출
        return mask

    def Find_RGB_Data(self) -> list:
        self.B_channel_data = self.image[:,:,0]
        self.G_channel_data = self.image[:,:,1]
        self.R_channel_data = self.image[:,:,2]
        return [self.B_channel_data, self.G_channel_data, self.R_channel_data]
    
    def Filter_RGB_Data(self, channel_data: np.ndarray) -> np.ndarray:
        filter_channel_data = channel_data[(channel_data != 0) & (channel_data >= 50)]
        filter_channel_count = np.bincount(filter_channel_data)
        
        valid_indices = np.where((filter_channel_count >= self.min_count)
                                &(filter_channel_count <= self.max_count))[0]
        
        sorted_indices = np.argsort(filter_channel_count[valid_indices])[::-1]  # 빈도수 높은 순서로 정렬된 값의 인덱스
        highest_indices = valid_indices[sorted_indices][:self.rank]
        return highest_indices
    
    def RGB_Mask(self, channel_data: np.ndarray, channel_value: int) -> np.ndarray:
        image = copy.deepcopy(self.image)
        condition = image[:,:,channel_value] <= np.average(channel_data)*0.85
        image[condition] = [255,255,255]
        image[~condition] = [0,0,0]
        return image
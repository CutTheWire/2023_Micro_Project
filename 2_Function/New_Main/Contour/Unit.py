import cv2
import numpy as np
import copy

from matplotlib import pyplot as plt

from IMG.IPP import ImageCV

class Seed:
    def __init__(self, frame) -> None:
        self.IC = ImageCV()
        self.image = frame
        self.colors = {
            "contour": (0,0,255),
            "text": (255,255,255)
        }
        
        # def Find_Contours
        self.threshold_area = 150

        # def Find_RGB_Data
        self.R_channel_data = None
        self.G_channel_data = None
        self.B_channel_data = None

        # def Filter_RGB_Data
        self.min_count = 100
        self.max_count = 5000
        self.rank = 500

        # def divide_into_intervals
        self.interval_size = 10

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
    
    
    def contour_areas_avg(self, filtered_contours: np.ndarray) -> tuple[list, float, list]:
        '''
        Input -> 윤곽선 \n
        Output -> 윤곽선 그룹 평균, 상위 & 하위 10% 제외한 평균, 구간(50) 넓이
        '''
        contour_areas = []
        for contour in filtered_contours:
            contour_area = cv2.contourArea(contour)
            contour_areas.append(contour_area)

        # 50의 배수에 따라 그룹으로 나눔
        grouped_data = {}
        for value in contour_areas:
            group = int(value // 50) * 50
            if group in grouped_data:
                grouped_data[group].append(value)
            else:
                grouped_data[group] = [value]

        # 선택된 그룹들의 평균 계산
        averages = []
        values_in_group = grouped_data[group]
        group_average = np.mean(values_in_group)
        averages.append(group_average)

        # 그룹의 데이터를 평탄하게 추출
        all_group_values = [value for values in grouped_data.values() for value in values]

        # 데이터를 정렬한 후 상위 10%와 하위 10%에 해당하는 값을 제외한 나머지 값들을 선택
        sorted_values = sorted(all_group_values)
        total_values = len(sorted_values)
        top_10_percent = int(total_values * 0.1)
        bottom_10_percent = int(total_values * 0.9)
        middle_values = sorted_values[top_10_percent:total_values - bottom_10_percent]

        # 상위 10%와 하위 10%를 제외한 값들의 평균 계산
        filtered_average = np.mean(middle_values)
        return averages, filtered_average, contour_areas
    
    def divide_into_intervals(self, areas: list) -> dict:
        intervals = {}
        for area in areas:
            interval = int(area / self.interval_size)
            if interval in intervals:
                intervals[interval].append(area)
            else:
                intervals[interval] = [area]
        return intervals

    def Black_Contour(self, image: np.ndarray) -> np.ndarray:
        # 이미지를 그레이스케일로 변환
        gray = self.IC.gray(image)

        # 이진화 처리 (검은색이 아닌 부분만을 추출하기 위함)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # 윤곽선 검출
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area = self.Area_Contours(contours)

        # 원본 이미지에 윤곽선 그리기
        result = image.copy()
        cv2.drawContours(result, contours, -1, self.colors["contour"], 2)  # 빨간색으로 윤곽선 그리기, 두 번째 인자는 윤곽선 목록

        # 단위로 구간을 나누고 각 구간의 넓이를 저장합니다.
        intervals = self.divide_into_intervals(area)

        # 가장 높은 갯수를 가진 구간을 찾습니다.
        max_interval = max(intervals, key=lambda k: len(intervals[k]))

        # 해당 구간의 넓이 평균을 계산합니다.
        if max_interval in intervals:
            max_interval_areas = intervals[max_interval]
            average_area = sum(max_interval_areas) / len(max_interval_areas)

        index = 0
        # 각 도형의 중심점에 번호 표시
        for contour, area in zip(contours, area):
            if area >= average_area*1.95:
                index += 2
                color = (255, 255, 255)

            elif  area <= average_area*0.2:
                continue
            else:
                index += 1
                color = (0, 155, 0)

            # 중심점 계산
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(result, str(index), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.putText(result, f"PART COUNT: {index}", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, self.colors["text"], 2)
        return result

    def Find_Contours(self, image: np.ndarray) -> np.ndarray:
        gray = self.IC.gray(image)
        # 이진화 처리 (검은색이 아닌 부분만을 추출하기 위함)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        # 윤곽선 검출
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # 윤곽선 면적을 계산하고 일정 크기 이하의 윤곽선을 필터링
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.threshold_area]
        contour_image = cv2.drawContours(self.image.copy(), filtered_contours, -1, self.colors["contour"], 2)
        return contour_image, filtered_contours
    
    def Area_Contours(self, contours: np.ndarray) -> np.ndarray:
        area = []
        for contour in contours:
            area.append(cv2.contourArea(contour))
        return area
    
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

        # 각 채널 데이터에서 0 값을 제외
        self.B_channel_data = np.where(self.B_channel_data != 0, self.B_channel_data, 255)
        self.G_channel_data = np.where(self.G_channel_data != 0, self.G_channel_data, 255)
        self.R_channel_data = np.where(self.R_channel_data != 0, self.R_channel_data, 255)

        return [self.B_channel_data, self.G_channel_data, self.R_channel_data]

    
    def Filter_RGB_Data(self, channel_data: np.ndarray) -> np.ndarray:
        filter_channel_data = channel_data[(channel_data != 0) & (channel_data >= 50)]
        filter_channel_count = np.bincount(filter_channel_data)
        
        valid_indices = np.where((filter_channel_count >= self.min_count)
                                &(filter_channel_count <= self.max_count))[0]
        
        sorted_indices = np.argsort(filter_channel_count[valid_indices])[::-1]  # 빈도수 높은 순서로 정렬된 값의 인덱스
        highest_indices = valid_indices[sorted_indices][:self.rank]
        return highest_indices
    
    def RGB_Mask(self, channel_data: np.ndarray, channel_value: int, x: float) -> np.ndarray:
        '''
        Input : 색상 채널, 채널 번호, 색상값 조정치\n
        Output : 마스크 이미지
        '''
        image = copy.deepcopy(self.image)
        condition = image[:,:,channel_value] <= np.average(channel_data)*x
        image[~condition] = [0,0,0]
        return image
    
    

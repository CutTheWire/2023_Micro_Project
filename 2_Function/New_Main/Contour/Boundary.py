import cv2
import numpy as np

class BoundaryContour:
    def __init__(self, frame: np.ndarray) -> None:
        self.image = frame

        # def return_color
        self.colors = {
            "blue": (227, 55, 55),
            "green": (182, 55, 78),
            "purple": (39, 217, 144),
            "other" : (55, 55, 55),
        }

        # def return_range_color
        self.range_colors = {
            #HSV값 [색상(H), 채도(S), 명도(V)]
            "green": (np.array([40, 80, 80]), np.array([90, 255, 255])), #녹색 검출
            "purple": (np.array([160, 80, 80]), np.array([175, 255, 255])), # 보라 또는 짙은 핑크와 같은 적색 계열 검출
            "other" : (np.array([0, 0, 0]), np.array([0, 0, 80])) # return_range_color의 color 선언 확인
        }
        # def Find_Contours, def draw_Contour
        self.image = frame

    def return_color(self, color: str):
        if color not in self.colors:
            color = "other"
        return self.colors.get(color, self.colors[color])

    def return_range_color(self, color: str):
        if color not in self.range_colors:
            color = "other"
        lower, upper = self.range_colors.get(color, self.range_colors[color])
        return lower, upper

    def Find_Contours(self, color: str) -> np.ndarray:
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower, upper = self.return_range_color(color)
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        return contours
        
    def Filter_Contours(self, contours: np.ndarray) -> dict:
        contour_dict = {}
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                contour_dict[area] = contour
        return contour_dict
    
    def draw_Contour(self, Contours_list: list, color: str) -> np.ndarray:
        color = color.lower()
        if color ==  "green":
            draw_size = 10
        elif color == "purple":
            draw_size = 5
        else:
            draw_size = 5
            
        for contour in Contours_list:
            cv2.drawContours(self.image , [contour], -1, self.return_color(color), draw_size)
        return self.image
    
    def FitLine_Contours(self, contours: np.ndarray, image: np.ndarray) -> (np.ndarray, list):
        # cv2.fitLine() 함수를 사용하여 최소제곱 직선 추정
        [vx, vy, x, y] = cv2.fitLine(contours, cv2.DIST_L2, 0, 0.01, 0.01)

        # 직선의 방향 벡터 (vx, vy)를 사용하여 1차 방정식 계수 구하기
        m = vy / vx
        b = y - (m * x)

        width = image.shape[1]

        # 직선의 방향 벡터 (vx, vy)를 사용하여 두 점을 계산합니다.
        lefty = int((-x * vy / vx) + y)
        righty = int(((width - x) * vy / vx) + y)

        # 추정된 직선을 이미지에 그립니다.
        cv2.line(image, (width - 1, righty), (0, lefty), self.return_color("blue"), 5)

        # 그린 이미지를 반환합니다.
        return image, [m, b]
    
    def Intersection_Pos(self, contour1: np.ndarray, contour2: np.ndarray, image: np.ndarray) -> np.ndarray:
        intersection_points = []

        for point1 in contour1:
            for point2 in contour2:
                if np.array_equal(point1, point2):
                    intersection_points.append(tuple(point1[0]))

        # 겹치는 점을 표시합니다.
        for point in intersection_points:
            cv2.circle(image, point, 5, (255, 255, 255), -1)
        return image
    
    @property
    def image(self):
        return self._image
    
    @image.setter
    def image(self, frame):
        self._image = frame


    
class BoundaryPos:
    def __init__(self) -> None:
        self.epsilon_ratio = 0.04
        self.fix_pos = None
        
    def Approximate_Pos(self, contours: np.ndarray) -> np.ndarray:
        epsilon = self.epsilon_ratio * cv2.arcLength(contours, True)
        return cv2.approxPolyDP(contours, epsilon, True)

    def Find_Nearest_Pos(self, main_pos: np.ndarray, sub_pos: np.ndarray) -> np.ndarray:
        for sub_coord in sub_pos:
            min_distance = float('inf')
            closest_index = None
            
            for i in range(len(main_pos)):
                distance = np.linalg.norm(main_pos[i] - sub_coord)
                if distance < min_distance:
                    min_distance = distance
                    closest_index = i
            
            # 가장 근사한 좌표 대체
            main_pos[closest_index] = sub_coord
        if len(main_pos) != 4:
            ValueError("len(Pos) is not 4")
        return main_pos
    
    def Find_Equation_Pos(self, equation_list: np.ndarray, equation_contour: np.ndarray) -> np.array:
        # 직선의 방정식
        a = equation_list[0][0]
        b = equation_list[1][0]

        # contour 데이터의 x, y 좌표를 분리
        x_contour = equation_contour[:, 0, 0]

        # x 좌표 중에서 가장 작은 값과 가장 큰 값을 찾음
        min_x = min(x_contour)
        max_x = max(x_contour)

        # 직선 함수에 x 좌표 대입하여 y 좌표 계산
        y_min_x = a * min_x + b
        y_max_x = a * max_x + b
        return np.array([[[max_x, y_max_x]], [[min_x, y_min_x]]])

        

class FixPos:
    def __init__(self, pos) -> None:
        self.location = None  # 먼저 self.location을 초기화합니다.
        self.fix_pos = pos  # 그 다음에 self.fix_pos를 설정합니다.

        # def return_color
        self.colors = {
            "red": (55, 55, 227),
            "blue": (227, 55, 55),
            "green": (55, 227, 55),
            "other":  (55, 55, 55)
        }

    @property
    def fix_pos(self):
        return self._fix_pos

    @fix_pos.setter
    def fix_pos(self, pos: np.ndarray) -> np.ndarray:
        location_nparr = []
        for corner in pos:
            x, y = corner[0]
            location_nparr.append([x, y])
        
        location2 = np.array([ [0, 1200],[900, 1200], [900, 0], [0, 0]], np.float32)

        self.location = np.array(location_nparr, np.float32)
        if len(self.location) == 4:
            self._fix_pos = cv2.getPerspectiveTransform(self.location, location2)
        else:
            self._fix_pos = self.location

    def return_color(self, color: str):
        if color not in self.colors:
            color = "other"
        return self.colors.get(color, self.colors[color])
        
    def apply_transform(self, image: np.ndarray, color: str) -> np.ndarray:
        color = color.lower()
        if self.location is not None:
            for point in self.location:
                x, y = point
                draw_color = self.return_color(color)  # 빨간색 원을 그립니다.
                cv2.circle(image, (int(x), int(y)), 15, draw_color, -1)  # 원을 그립니다
        else:
            pass
        return image


'''
-------------------------------------------테스트-------------------------------------------
'''

# if __name__ == "__main__":
#     image = cv2.imread("C:/tinywave_2/RGB_3.png")
#     BC = BoundaryContour(image)
#     contours = BC.Find_Contours(BC.lower_green, BC.upper_green)
#     # 컨투어 그리기 (내부 윤곽선만)
#     image = BC.Contour_draw(BC.Green_color, contours)

#     cv2.imshow("", image)
#     cv2.waitKey(0)
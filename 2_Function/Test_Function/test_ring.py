import cv2
import numpy as np

# 이미지를 읽어오기
image = cv2.imread("./test1.png")

# 이미지를 그레이스케일로 변환
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 검은색 영역 찾기 (검은색은 값이 0)
_, thresh = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)

# 검은색 영역에서 윤곽선 찾기
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 윤곽선을 원본 이미지에 그리기
result_image = np.copy(image)
cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)  # 윤곽선을 초록색으로 그리기

# 각 윤곽선의 넓이와 중앙 좌표 계산 및 출력
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])  # 중심의 x 좌표
        cy = int(M["m01"] / M["m00"])  # 중심의 y 좌표

        # 중앙 상단에 번호 그리기
        cv2.putText(result_image, str(i + 1), (cx - 10, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        print(f"윤곽선 {i + 1} - 넓이: {area}, 중앙 좌표: ({cx}, {cy})")

# 결과 이미지 출력
cv2.imshow("Contours on Black Area with Numbers", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

# 파란색 범위 설정 (BGR 순서로 설정)
lower_blue = np.array([80, 40, 40])
upper_blue = np.array([130, 255, 255])

# 카메라 열기
cap = cv2.VideoCapture(0)  # 0은 기본 카메라 장치 번호, 더 많은 카메라가 있는 경우에는 1, 2 등을 시도해볼 수 있습니다.

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # BGR을 HSV로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 파란색 범위 내의 영역 찾기
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 컨투어 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어 그리기
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # 이미지 출력
    cv2.imshow('Blue Contours', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 종료
cap.release()
cv2.destroyAllWindows()

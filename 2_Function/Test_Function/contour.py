import cv2
import numpy as np
import copy
import image_save as IS

# 이미지를 읽어옵니다.
image = cv2.imread('./1.jpg')

# 그레이스케일로 변환합니다.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 이진화를 수행합니다. 밝기가 100 이상인 픽셀을 하얀색(255)으로 설정합니다.
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# thresh 이미지를 색 반전합니다.
thresh = cv2.bitwise_not(thresh)
# 윤곽선을 찾습니다.
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 원본 이미지에 윤곽선을 그립니다.
result_image = copy.deepcopy(image)

# 밝기가 50 이상인 픽셀을 하얀색으로 칠한 이미지를 만듭니다.
result_image [thresh == 0] = [255, 255, 255]
cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)  # 윤곽선을 초록색으로 그립니다.

# 윤곽선 수를 출력합니다.
contour_count = len(contours)

# 각 윤곽선 중심에 카운트를 그립니다.
font = cv2.FONT_HERSHEY_SIMPLEX
for i, contour in enumerate(contours):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(result_image, str(i + 1), (cX, cY), font, 1, (0, 0, 255), 2)


cv2.putText(result_image, f"seed count: {contour_count}", (10, 30), font, 1, (0, 0, 255), 2)

# 결과 이미지를 표시합니다.
cv2.imshow('Contours', result_image)
IS.image_save(result_image, " ")
# 키 입력 대기 및 윈도우 종료
cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
import cv2


def hsv():
    BGR = np.uint8([[[157, 153, 164]]])

    hsv_ = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)

    print('HSV for BGR', hsv_)


hsv()
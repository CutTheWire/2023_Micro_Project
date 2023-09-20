# Image Pre-Processing
import cv2
import numpy as np

class ImageCV:
    def __init__(self) -> None:
        self.brigtness = -30 #양수 밝게, 음수 어둡게
        self.alp = 1.0
        self.kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.kernel_ones = np.ones((3,3),np.uint8)
        

    def Image_Crop(self, image, pos: np.ndarray, dsize: tuple) -> np.ndarray:
        dst = cv2.warpPerspective(image, pos, dsize)
        return cv2.flip(dst, 0)
    
    def Scale_Resolution(self, image: np.ndarray, Scale: float) -> tuple:
        height, width = image.shape[:2]
        return (int(width*Scale), int(height*Scale))
    
    def Image_Slice(self, image: np.ndarray, height_value: float, width_value: float) -> np.ndarray:
        height, width = image.shape[:2]
        fix_value = [0,0]
        values = [height_value, width_value]

        for i in range(len(values)):
            if values[i] <= 0.01 and values[i] > 0:
                fix_value[i] = 0

            elif values[i] > 0.01:
                fix_value[i] = 0.01

            elif values[i] <= 0:
                values[i] = 0
                fix_value[i] = 0
            
        return image[int(height*values[0]):int(height*(1-values[0]+fix_value[0])):,
                    int(width*values[1]):int(width*(1-values[1]+fix_value[1]))]
    
    def Brightness(self, image: np.ndarray) -> np.ndarray:
        image = (np.int16(image) + self.brigtness).astype(np.uint8)
        return image
    
    def Dilate(self, image: np.ndarray) -> np.ndarray:
        image = np.clip((1.0+self.alp) * image - 128 * self.alp, 0, 255)
        image = cv2.dilate(image,self.kernel_5, iterations = 1)
        return image
    
    def Binarization(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        return

    def Pos_by_Img(self, image, pos):
        image = self.Image_Crop(image, pos, (900,1200))
        image = self.Image_Slice(image, height_value=0.02, width_value=0.02)
        image = self.Brightness(image)

        result_image_Resolution = self.Scale_Resolution(image, 0.5)
        image = cv2.resize(image, result_image_Resolution)
        return image

    

'''
-------------------------------------------테스트-------------------------------------------
'''

# if __name__ == "__main__":
#     image = cv2.imread("C:/Users/sjmbe/TW/TEST/230908/161906_Micro.jpg")
#     IC = ImageCV()
#     cv2.imshow("test",IC.Binarization(IC.Brightness(image)))
#     cv2.waitKey(0)

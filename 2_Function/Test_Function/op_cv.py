import cv2
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.segmentation import watershed
import numpy as np

# Load the image
image = cv2.imread("C:/Users/sjmbe/TW/Micro/230825/1.jpg")

# Apply mean shift filtering
shifted = cv2.pyrMeanShiftFiltering(image, 52, 30)

# Display the original input image
cv2.imshow("Input", image)

# Convert the mean shift image to grayscale and apply Otsu's thresholding
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)

# Compute the exact Euclidean distance from every binary pixel to the nearest zero pixel
D = ndimage.distance_transform_edt(thresh)

# Find local maxima in the distance map
localMax = peak_local_max(D, min_distance=20, labels=thresh)

# Perform connected component analysis on the local maxima
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]

# Apply Watershed algorithm using the modified code
labels = watershed(-D, markers, mask=thresh)

# Count the number of unique segments
num_segments = len(np.unique(labels)) - 1
print("[INFO] {} unique segments found".format(num_segments))

# Display the labeled image
cv2.imshow("Segmented Image", labels)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

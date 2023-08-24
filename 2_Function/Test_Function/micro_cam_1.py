import cv2

# Read the color image
image = cv2.imread("./WIN_20230801_13_08_40_Pro.jpg")

# Resize the image to a smaller size for display
height, width = image.shape[:2]
new_height = int(height * 0.5)  # Resize to 50% of the original height
new_width = int(width * 0.5)    # Resize to 50% of the original width
resized_image = cv2.resize(image, (new_width, new_height))

# Make a copy
new_image = resized_image.copy()

# Convert the image to grayscale
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to binary
ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

# Invert the binary image
inverted_binary = ~binary

# Find the contours on the inverted binary image
contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours (in red) on the original image and display the result
with_contours = cv2.drawContours(resized_image, contours, -1, (255, 0, 255), 1)
cv2.imshow('Detected contours', with_contours)
cv2.waitKey(0)

# Draw just the first contour
first_contour = cv2.drawContours(new_image, contours, 0, (255, 0, 255), 1)
cv2.imshow('First detected contour', first_contour)
cv2.waitKey(0)

# Draw a bounding box around the first contour
x, y, w, h = cv2.boundingRect(contours[0])
cv2.rectangle(first_contour, (x, y), (x + w, y + h), (255, 0, 0), 1)
cv2.imshow('First contour with bounding box', first_contour)
cv2.waitKey(0)

# Draw a bounding box around all contours
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    # Make sure contour area is large enough
    if cv2.contourArea(c) > 10:
        cv2.rectangle(with_contours, (x, y), (x + w, y + h), (255, 0, 0), 1)

cv2.imshow('All contours with bounding box', with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()

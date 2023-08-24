import cv2

import tkinter_image as TI

def detect_objects():
    ret, frame = cap.read()

    frame = cv2.add(frame, 50)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < area_threshold:
            continue

        object_count += 1

        x, y, w, h = cv2.boundingRect(contour)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.drawContours(frame, [contour], 0, (255, 0, 255), 2)

    font_scale = 1.0
    text_thickness = 2
    text_org = (10, 30)
    cv2.putText(frame, "Total Count: " + str(object_count), text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 255), text_thickness)
    frame = cv2.resize(frame, (800,800), interpolation=cv2.INTER_LANCZOS4)
    image = TI.cv2_to_tkinter_image(frame)
    label.configure(image=image)
    label.image = image

    window.after(10, detect_objects)
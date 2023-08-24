import cv2
import darknet

def load_pretrained_model():
    # YOLO 설정 파일과 가중치 파일 로드
    net, class_names, _ = darknet.load_network("cfg/yolov3.cfg", "yolov3.weights", "data/coco.names")
    return net, class_names

def detect_objects(image, net, class_names):
    # 이미지를 YOLO 입력 크기로 리사이징
    height, width, _ = image.shape
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image.tobytes())

    # 객체 검출 수행
    detections = darknet.detect_image(net, class_names, darknet_image)

    # 검출된 객체들의 위치 리스트 반환
    detected_seeds = [(int(x), int(y)) for _, _, (x, y, _, _) in detections]

    return detected_seeds

if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    image = cv2.imread(image_path)

    # 객체 검출을 위한 미리 학습된 모델 로드
    net, class_names = load_pretrained_model()

    # 객체 검출 수행
    detected_seeds = detect_objects(image, net, class_names)

    # 씨앗 카운팅
    seed_count = len(detected_seeds)

    # 결과 시각화 (원 그리기)
    for (x, y) in detected_seeds:
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    # 결과 출력
    cv2.putText(image, f"Seed Count: {seed_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Seed Count Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

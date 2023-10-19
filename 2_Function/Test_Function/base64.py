import base64

# 이미지 파일을 바이트로 읽기
with open("C:/Users/sjmbe/Downloads/SET.png", 'rb') as image_file:
    image_bytes = image_file.read()

# 이미지 바이트를 BASE64로 인코딩
image_base64 = base64.b64encode(image_bytes).decode('utf-8')

# 결과 출력
print(image_base64)

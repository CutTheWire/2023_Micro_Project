import sympy as sp

# 직선의 방정식
a = -0.006623352412134409
b = 268.556396484375

# 원의 방정식
h = 0  # 원의 중심 좌표
k = 0
r = 500  # 반지름

# 변수 정의
x, y = sp.symbols('x y')

# 직선과 원의 방정식 설정
equation1 = a * x + b - y
equation2 = (x - h)**2 + (y - k)**2 - r**2

# 교차점 계산
solutions = sp.solve((equation1, equation2), (x, y))

# 교차점 출력
for solution in solutions:
    x_val, y_val = solution
    print(f"교차점: x = {x_val}, y = {y_val}")

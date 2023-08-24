import numpy as np

approx_pos = np.array([
    [[1597, 163]],
    [[1629, 1189]],
    [[930, 1210]],
    [[887, 178]]
])

pupple_pos = np.array([
    [[1601, 295]],
    [[897, 322]]
])

def find_nearest_approx(pupple_pos, approx_pos):
    # 각 Pupple_pos 좌표에 대해 가장 근사한 좌표 찾기
    for pupple_coord in pupple_pos:
        min_distance = float('inf')
        closest_index = None
        
        for i in range(len(approx_pos)):
            distance = np.linalg.norm(approx_pos[i] - pupple_coord)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        # 가장 근사한 좌표 대체
        approx_pos[closest_index] = pupple_coord
    return approx_pos

print("대체된 배열:")
print(find_nearest_approx(pupple_pos, approx_pos))

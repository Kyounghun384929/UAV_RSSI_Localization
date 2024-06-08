import numpy as np
import json
import os

# 초기 스캔 이동 위치 설정
def Init_ScanWayPoint(Lx: int, Ly: int, Cx:int , Cy:int) -> list:
    Snodes = []
    x, y = 0, 0
    
    while x <= Lx:
        temp = y
        while y <= Ly:
            Snodes.append((x, y))
            y += 2 * Cy
        if temp == 0:
            y = Cy
        else:
            y = 0
        x += Cx
    
    return Snodes

# 지상 물체 스캔 이동 위치 설정
def CellWayPoint(Lx, Ly, Cx, Cy, w=2):
    Cnodes = []
    x = Cx / (2 * w)
    y = Cy / (2 * w)
    
    while x <= Lx:
        while y <= Ly:
            Cnodes.append((x, y))
            y = y + Cy / w
        y = Cy / (2 * w)
        x = x + Cx / w
        
    return Cnodes

# 지상 물체 생성 및 저장
def set_gobjects(Lx, Ly, num_objects=30):
    # file_path = './Environ/ground_objects.json'
    
    # if os.path.isfile(file_path):
    #     with open(file_path, 'r') as file:
    #         ground_objects = json.load(file)
    # else:
    #     ground_objects = []
    #     for i in range(num_objects):
    #         object = {
    #             'id': i,
    #             'x': round(np.random.uniform(0, Lx)),
    #             'y': round(np.random.uniform(0, Ly))
    #         }
    #         ground_objects.append(object)

    # with open(file_path, 'w') as file:
    #     json.dump(ground_objects, file, indent=4)
    
    ground_objects = []
    for i in range(num_objects):
        object = {
            'id': i,
            'x': round(np.random.uniform(0, Lx)),
            'y': round(np.random.uniform(0, Ly)),
            'vx': round(np.random.normal(0, 1.5)),
            'vy': round(np.random.normal(0, 1.5))
        }
        ground_objects.append(object)
    
    return ground_objects
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    Lx = 900        # 가로 [m]
    Ly = 700        # 세로 [m]
    h = 100         # 고도 [m]
    D = 200         # 통신반경 [m]
    w = 2           # Cell 나누는 단위

    R = np.sqrt(D**2 - h**2)
    Cx = Lx / np.ceil(Lx/R)
    Cy = Ly / np.ceil(Ly/R)
    
    scan_points = Init_ScanWayPoint(Lx, Ly, Cx, Cy)
    cell_points = CellWayPoint(Lx, Ly, Cx, Cy, w)
    ground_objects = set_gobjects(Lx, Ly, 30)
    
    plt.figure(figsize=(10, 7))
    plt.grid(False)

    # 초기 스캔 지점 Plot
    for i, point in enumerate(scan_points):
        if i == 0:  # 첫 번째 요소에만 라벨을 추가
            plt.plot(point[0], point[1], 'o', markersize=10, c='#FF0C0C', label='Initial Scan Point')
        else:
            plt.plot(point[0], point[1], 'o', markersize=10, c='#FF0C0C')
        plt.axhline(y=point[1], color='lightgray', linestyle='--')
        plt.axvline(x=point[0], color='lightgray', linestyle='--')
        
        # 반경 200m 원 추가
        circle = plt.Circle((point[0], point[1]), 200, color='#007BFF', fill=False, linestyle='--')
        plt.gca().add_patch(circle)

    # 셀 지점 Plot
    for i, point in enumerate(cell_points):
        if i == 0:  # 첫 번째 요소에만 라벨을 추가
            plt.plot(point[0], point[1], '^', markersize=7, c='#0011FF', label='Way Point')
        else:
            plt.plot(point[0], point[1], '^', markersize=7, c='#0011FF')

    # 지상 물체 위치 Plot
    for i, object in enumerate(ground_objects):
        if i == 0:  # 첫 번째 요소에만 라벨을 추가
            plt.plot(object['x'], object['y'], '*', markersize=10, c='#000000', label='Ground Object Truth Position')
        else:
            plt.plot(object['x'], object['y'], '*', markersize=10, c='#000000')
    
    # plt.ylim(-200, 900)
    plt.legend(loc='center', bbox_to_anchor=(1, 1), shadow=True)
    plt.tight_layout()
    plt.show()
from .UAVChannelModel import *
import numpy as np

def trilateration(pos1: list[float, float, float], pos2: list, pos3: list, n: float) -> list[float, float]:
    '''
    3개의 x, y 좌표 및 RSSI 신호 강도를 이용하여, RSSI 측정 신호 기반 거리를 계산하고, 그 거리를 이용하여 estimate position을 반환함.

    Returns:
    추정된 Ground Object의 위치
    [x, y]
    '''
    d1 = estimate_distance(pos1[2], n)
    d2 = estimate_distance(pos2[2], n)
    d3 = estimate_distance(pos3[2], n)
    
    x1, y1 = pos1[0], pos1[1]
    x2, y2 = pos2[0], pos2[1]
    x3, y3 = pos3[0], pos3[1]
    
    # 삼변측량 계산
    A = np.array([
        [2 * (x1 - x3), 2 * (y1 - y3)],
        [2 * (x2 - x3), 2 * (y2 - y3)]
    ])
    B = np.array([
        x1**2 - x3**2 + y1**2 - y3**2 + d3**2 - d1**2,
        x2**2 - x3**2 + y2**2 - y3**2 + d3**2 - d2**2
    ])
    
    try:
        X = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        if rank < 2:
            return [0., 0.]
    return X[0], X[1]
    
def multilateration(positions, n):
    '''
    여러 기준점의 위치와 RSSI 값을 기반으로 다변측량을 수행하여 위치를 추정
    Args:
        positions (list): [(x1, y1), (x2, y2), ...] 형태의 기준점 위치 리스트
        rssis (list): [RSSI1, RSSI2, ...] 형태의 RSSI 값 리스트
        
    Returns:
        tuple: 추정된 위치 (x, y)
    '''
    A = []
    B = []

    for pos in positions:
        x, y, rssi = pos
        d = estimate_distance(rssi, n)  # RSSI를 거리로 변환

        A.append([2 * x, 2 * y])
        B.append(x**2 + y**2 - d**2)

    A = np.array(A)
    B = np.array(B)

    # 최소자승법으로 위치 추정
    X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    return X[0], X[1]
    
if __name__ == "__main__":
    # 예시 사용
    positions = [(0, 0), (10, 0), (5, 8), (3, 5)]  # 기준점 위치
    rssis = [-55, -60, -62, -58]                   # RSSI 값
    
    trilateration(positions[0], positions[1], positions[2], n=3)
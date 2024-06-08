import numpy as np
from scipy.optimize import least_squares

import sys
sys.path.append('c:/Users/cai_kkh/OneDrive/KH Library/Code/0.Python/1.Project/UAVLocalizationRSSI(GraduateWork)')
from ChannelModel.revised_channelmodel import *


def estimate_position(rssi_measurements, waypoints):
    # RSSI 측정값과 UAV의 웨이포인트를 사용하여 객체의 위치를 추정합니다.
    distances = rssi_to_distance(rssi_measurements)
    estimated_position = multilateration(distances, waypoints)
    return estimated_position

def rssi_to_distance(rssi_measurements):
    # RSSI 측정값을 거리로 변환합니다.
    distances = []
    for rssi in rssi_measurements:
        rssi_dbm = rssi_to_dbm(rssi)
        distance = 10 ** ((27.55 - (20 * np.log10(frequency)) - rssi_dbm) / 20)
        distances.append(distance)
    return distances

def rssi_to_dbm(rssi):
    # RSSI 값을 dBm으로 변환합니다.
    return rssi - 100

def multilateration(distances, waypoints):
    # Multilateration 알고리즘을 사용하여 객체의 위치를 추정합니다.
    def equations(variables):
        x, y = variables
        return [(x - wp[0])**2 + (y - wp[1])**2 - d**2 for wp, d in zip(waypoints, distances)]
    
    initial_guess = np.mean(waypoints, axis=0)
    result = least_squares(equations, initial_guess)
    return result.x

# 사용할 주파수 (MHz 단위)
frequency = 2400

# 예시 웨이포인트와 RSSI 측정값
waypoints = [(100, 100), (150, 80), (120, 120)]
rssi_measurements = [70, 75, 65]

# 실제 목표 위치
actual_position = (125, 95)

# 객체의 위치 추정
estimated_position = estimate_position(rssi_measurements, waypoints)

# 추정 위치와 실제 위치의 차이 계산
position_error = np.linalg.norm(np.array(estimated_position) - np.array(actual_position))

# Localization Error를 미터(m) 단위로 변환 (1 unit = 1 meter로 가정)
position_error_meters = position_error * 1  # 여기서 1 unit은 1 meter에 해당합니다.

print(f"Estimated Position: {estimated_position}")
print(f"Actual Position: {actual_position}")
print(f"Localization Error: {position_error_meters:.12f} m")

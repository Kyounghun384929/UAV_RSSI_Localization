import numpy as np
import matplotlib.pyplot as plt
# import sys
# sys.path.append('c:/Users/cai_kkh/OneDrive/KH Library/Code/0.Python/1.Project/UAVLocalizationRSSI(GraduateWork)')
# from ChannelModel.revised_channelmodel import *

frequency = 2e9 # 논문 2GHz 기준
c = 3e8 # 빛 속도 m/s
environ_params = {
    'aLoS': 10, 'bLoS': 2, 'aNLoS': 30, 'bNLoS': 1.7, 'a0' : 47, 'b0' : 20
    }

def cal_distance(r: int, h: int) -> int:
    '''수평거리, 높이를 고려한 직선 거리 반환'''
    return np.sqrt(r**2 + h**2)

def FSPL(r: int, h: int) -> int:
    '''
    args:
    r = 수평거리
    h = UAV 높이
    
    returns:
    FSPL dB Value (양수)
    '''
    d = cal_distance(r, h)
    return 20 * np.log10(d) + 20 * np.log10(frequency) + 20 * np.log10(4 * np.pi / c)

def LogNormal_proposed_ver2(r: int, h: int):
    '''
    논문의 모델을 보다 현실적으로 수정함.
    환경에 대한 사전 지식이 없는 경우 더 유용할 수 있음.
    거리가 멀어질 수록 NLos 확률이 높게 FSPL 보다 더 낮은 RSSI 값을 가지도록 유도함.
    '''
    if r == 0:
        theta = np.pi / 2
    else:
        theta = np.arctan(h / r)
    
    P_Los = 1 / (1 + environ_params['a0'] * np.exp(-environ_params['b0'] * theta))
    sigma_los = environ_params['aLoS'] * np.exp(-environ_params['bLoS'] * theta)
    sigma_nlos = environ_params['aNLoS'] * np.exp(-environ_params['bNLoS'] * theta)
    
    sigma = P_Los**2 * sigma_los**2 + (1 - P_Los)**2 * sigma_nlos**2
    
    fspl = FSPL(r, h)
    loss = np.random.normal(np.sqrt(sigma), np.sqrt(sigma))
    path_loss = fspl+loss
    return P_Los, path_loss

# Kalman Filter 클래스 정의
class KalmanFilter():
    def __init__(self, processNoise=0.005, measurementNoise=20):
        self.isInitialized = False
        self.processNoise = processNoise
        self.measurementNoise = measurementNoise
        self.predictedRSSI = 0
        self.errorCovarianceRSSI = 0
    
    def filtering(self, rssi):
        if not self.isInitialized:
            self.isInitialized = True
            priorRSSI = rssi
            priorErrorCovariance = 1
        else:
            priorRSSI = self.predictedRSSI
            priorErrorCovariance = self.errorCovarianceRSSI + self.processNoise
        
        kalmanGain = priorErrorCovariance / (priorErrorCovariance + self.measurementNoise)
        self.predictedRSSI = priorRSSI + (kalmanGain * (rssi - priorRSSI))
        self.errorCovarianceRSSI = (1 - kalmanGain) * priorErrorCovariance
        return self.predictedRSSI


if __name__ == '__main__':
    # h = 100  # UAV altitude
    # r = np.linspace(1, 300)
    # # _, RSSIs = path_loss(r, h)
    # _ = [FSPL(d, h) for d in r]
    # RSSIs = [LogNormal_proposed_ver2(d, h)[1] for d in r]
    
    # kalman = KalmanFilter()
    
    # kalman_RSSIs = [kalman.filtering(RSSI) for RSSI in RSSIs]
    
    # plt.figure()
    # plt.plot(RSSIs, label='Original RSSIs')
    # plt.plot(kalman_RSSIs, linestyle='--', c='#FF0000', label='Kalman RSSIs')
    # plt.plot(_, label='FSPL', c='#000000', linestyle='-')
    # plt.legend()
    # plt.show()
    
    # 시각화를 위한 파라미터 설정
    num_samples = 200  # 샘플 개수
    uav_height = 50  # UAV 높이 (m)
    max_distance = 300  # 최대 수평 거리 (m)

    # 랜덤 위치 생성 및 RSSI 계산
    np.random.seed(42)  # 재현성을 위한 시드 설정
    distances = np.random.randint(100, max_distance, num_samples)  # 랜덤 수평 거리 생성
    _, raw_rssi = zip(*[LogNormal_proposed_ver2(r, uav_height) for r in distances])  # RSSI 계산

    # 칼만 필터 적용
    kf = KalmanFilter()
    filtered_rssi = [kf.filtering(rssi) for rssi in raw_rssi]

    # 시각화
    plt.figure(figsize=(12, 6))

    # 랜덤 위치 분포 시각화
    plt.subplot(1, 2, 1)
    plt.scatter(distances, raw_rssi, label='Raw RSSI', alpha=0.7)
    plt.title('Random Location Distribution and Raw RSSI')
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('RSSI (dB)')
    plt.legend()

    # RSSI 시계열 및 칼만 필터 결과 시각화
    plt.subplot(1, 2, 2)
    plt.plot(raw_rssi, label='Raw RSSI', alpha=0.7)
    plt.plot(filtered_rssi, label='Filtered RSSI', color='orange')
    plt.title('RSSI Time Series and Kalman Filter Results')
    plt.xlabel('Sample Index')
    plt.ylabel('RSSI (dB)')
    plt.legend()

    plt.tight_layout()
    plt.show()



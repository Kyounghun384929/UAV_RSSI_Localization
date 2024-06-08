import numpy as np
import matplotlib.pyplot as plt

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

frequency = 2.4e9
speed_of_light = 3e8
lambda_f = speed_of_light / frequency

params = {'a0' : 45,
        'b0' : 10,
        'alos' : 10,
        'blos' : 2,
        'anlos' : 30,
        'bnlos': 1.7}

def FSPL(d):
    return 20 * np.log10(4 * np.pi * d / lambda_f)

def Log_Normal(d, n=4):
    return FSPL(d) + np.random.normal(0, 100)

def Air_to_Ground(r, h):
    theta = np.arctan(h/r)
    slos = params['alos'] * np.exp(-params['blos'] * theta)
    snlos = params['anlos'] * np.exp(-params['bnlos'] * theta)
    plos = 1 / (1 + params['a0'] * np.exp(-params['b0'] * (theta)))
    
    sigma = plos**2 * slos**2 + (1-plos)**2 * snlos**2
    noise = np.random.normal(0, sigma)
    return noise

def log_distance(d, noise):
    d0 = 1
    PL0 = FSPL(d0)
    n = 4
    
    return PL0 + 10 * n * np.log10(d/d0) + noise

if __name__ == '__main__':
    # d = np.linspace(1, 10000, 10000)
    # fspl = FSPL(d)
    # pl = Log_Normal(d)
    
    # plt.figure()
    # plt.plot(d, -fspl)
    # plt.plot(d, -pl)
    # plt.show()

    import numpy as np
    import matplotlib.pyplot as plt

    # Log-Normal Shadowing Path Loss Model 함수 정의
    def log_normal_shadowing_path_loss(fc, d, d0, n, sigma):
        """
        fc: 캐리어 주파수 (Hz)
        d: 기지국과 모바일 스테이션 사이의 거리 (m)
        d0: 참조 거리 (m)
        n: 경로 손실 지수
        sigma: 표준 편차 (dB)
        """
        # 빛의 속도 (m/s)
        c = 299792458
        # 파장 계산 (m)
        lamda = c / fc
        # 평균 Path Loss 계산 (dB)
        PL_mean = -20 * np.log10(lamda / (4 * np.pi * d0)) + 10 * n * np.log10(d / d0)
        # Log-Normal Shadowing 추가
        PL = PL_mean + sigma * np.random.randn(len(d))
        return PL

    # 파라미터 설정
    fc = 2.4e9  # 2.4 GHz
    d0 = 1  # 1 meter
    n = 3  # Urban cellular radio
    sigma = 4  # dB

    # 거리 범위 설정 (1m부터 1000m까지)
    distances = np.linspace(1, 1000, 1000)

    # Path Loss 계산
    path_loss = log_normal_shadowing_path_loss(fc, distances, d0, n, sigma)

    # 시각화
    plt.figure()
    plt.plot(distances, -FSPL(distances), label='FSPL')
    plt.plot(distances, -path_loss, label='Log-Normal Shadowing')
    plt.xlabel('Distance (m)')
    plt.ylabel('Path Loss (dB)')
    plt.title('Log-Normal Shadowing Channel Model')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(-FSPL(100))

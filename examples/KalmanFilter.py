import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('c:/Users/cai_kkh/OneDrive/KH Library/Code/0.Python/1.Project/UAVLocalizationRSSI(GraduateWork)')
from ChannelModel.revised_channelmodel import *

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
    h = 100  # UAV altitude
    r = np.linspace(1, 300)
    # _, RSSIs = path_loss(r, h)
    _ = [FSPL(d, h) for d in r]
    RSSIs = [LogNormal_proposed_ver2(d, h)[1] for d in r]
    
    kalman = KalmanFilter()
    
    kalman_RSSIs = [kalman.filtering(RSSI) for RSSI in RSSIs]
    
    plt.figure()
    plt.plot(RSSIs, label='Original RSSIs')
    plt.plot(kalman_RSSIs, linestyle='--', c='#FF0000', label='Kalman RSSIs')
    plt.plot(_, label='FSPL', c='#000000', linestyle='-')
    plt.legend()
    plt.show()



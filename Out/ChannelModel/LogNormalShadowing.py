import numpy as np
import matplotlib.pyplot as plt

'''
신호가 건물 내부 또는 인구 밀도가 높은 지역에서 원거리에서 발생하는 경로 손실을 예측하는 무선 전파 모델
'''

# Log-Normal Shadowing Model parameters
Pt = 0  # 송신된 신호의 전력 (dBm)
K = 1   # 특정 거리에서의 평균적인 이득
d0 = 1  # 참조 거리 (m)
gamma = 2  # 경로 손실 지수
sigma = 6  # 표준편차 (dB)

# 거리 범위 (m)
d = np.linspace(1, 1000)

# Log-Normal Shadowing Model
Pr = Pt + 10 * np.log10(K) - 10 * gamma * np.log10(d/d0) - np.random.normal(0, sigma, len(d))

# 이동평균 계산
window_size = 10  # 이동평균 윈도우 크기
Pr_moving_avg = np.convolve(Pr, np.ones(window_size)/window_size, mode='same')

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(d, -Pr, label='Log-Normal Shadowing Model')
plt.plot(d, -Pr_moving_avg, label='Log-Normal', linestyle='--', c='red')
plt.xlabel('Distance (m)')
plt.ylabel('Received Signal Strength (dBm)')
plt.title('Log-Normal Shadowing Model Simulation')
plt.legend()
plt.grid(True)
plt.show()

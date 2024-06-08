import numpy as np
import matplotlib.pyplot as plt

# 기지국 위치 설정
base_stations = np.array([[0, 0], [100, 0], [50, 86.6]])

# 수신기 위치 설정
receiver = np.array([50, 43.3])

# AWGN 생성 함수
def awgn(signal, snr):
    snr_linear = 10**(snr/10)
    power_signal = np.sum(signal**2) / len(signal)
    power_noise = power_signal / snr_linear
    noise = np.sqrt(power_noise) * np.random.randn(len(signal))
    return signal + noise

# ToA 신호 생성
toa_signals = np.linalg.norm(base_stations - receiver, axis=1)

# SNR 설정
snr = 10  # dB 단위

# ToA 신호에 AWGN 추가
toa_signals_with_noise = awgn(toa_signals, snr)

# 시각화
plt.figure(figsize=(10, 8))
plt.scatter(base_stations[:, 0], base_stations[:, 1], c='red', label='Base Station')
plt.scatter(receiver[0], receiver[1], c='blue', label='Anchor')
plt.title('Visualize base station and receiver locations and ToA signals')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()

# 각 기지국에서 수신기까지의 거리를 원으로 표시
for i, toa in enumerate(toa_signals_with_noise):
    circle = plt.Circle(base_stations[i], toa, color='g', fill=False, linestyle='--')
    plt.gca().add_artist(circle)

plt.show()

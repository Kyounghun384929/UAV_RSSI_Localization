import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from decimal import Decimal

# 송신기의 실제 위치
transmitter = np.array([Decimal('2'), Decimal('3')])

# 수신기의 위치
receivers = np.array([[Decimal('0'), Decimal('0')], [Decimal('4'), Decimal('0')], [Decimal('0'), Decimal('4')]])

# 송신기로부터 수신기까지의 거리 계산
distances = np.sqrt(np.sum((receivers - transmitter)**2, axis=1))

# 속도 (예: 음속)
speed_of_signal = Decimal('343')

# 신호 도착 시간 계산
arrival_times = distances / speed_of_signal

# TDoA 계산
tdoa = arrival_times - arrival_times[0]

# 시각화
plt.figure(figsize=(8, 8))
sns.scatterplot(x=receivers[:, 0], y=receivers[:, 1], s=100, color='red', label='Receivers')
sns.scatterplot(x=[transmitter[0]], y=[transmitter[1]], s=100, color='blue', label='Transmitter')

# 각 수신기 위치에 TDoA 표시
for i, (x, y) in enumerate(receivers):
    plt.text(x, y, f'TDoA: {tdoa[i]:.2f}', fontsize=12)

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('TDoA Visualization')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np

# 주어진 Air-to-Ground channel 모델 파라미터
a0 = 45
b0 = 10
alos = 10
blos = 2
anlos = 30
bnlos = 1.7
c = 3e8  # 빛의 속도
f = 2e9  # 주파수

# 유클리드 거리 계산 함수
def euclidean(uav_state, ground_state):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(uav_state, ground_state)))

# 채널 손실 계산 함수
def channelLoss(uav_state, ground_state):
    r = euclidean(uav_state[:2], ground_state[:2])
    theta = np.arctan(uav_state[2] / r)
    d = euclidean(uav_state, ground_state)
    sigma_los = alos * np.exp(-blos * theta)
    sigma_nlos = anlos * np.exp(-bnlos * theta)
    prob_los = 1 / (1 + a0 * np.exp(-b0 * theta))
    covar = prob_los**2 * sigma_los**2 + (1 - prob_los)**2 * sigma_nlos**2
    np.random.seed(0)  # 고정 시드
    psai = np.random.normal(0, np.sqrt(covar))
    return -(20 * np.log10(d) + 20 * np.log10(4 * np.pi * f / c) + psai)

# RSSI 신호를 거리로 역산하는 함수
def rssiToDistance(rssi, uav_state, ground_state):
    d = 10 ** ((-rssi - 20 * np.log10(4 * np.pi * f / c)) / 20)
    return d

# 실제 위치, 추정 위치, 거리 오차를 출력하는 함수
def printLocationError(uav_state, ground_state, aps):
    actual_distances = [euclidean(uav_state, (ap.x, ap.y, 0)) for ap in aps]
    rssi_values = [channelLoss(uav_state, (ap.x, ap.y, 0)) for ap in aps]
    for ap in aps:
        estimated_distances = [rssiToDistance(rssi, uav_state, (ap.x, ap.y, 0)) for rssi in rssi_values]
        
        # Multilateration 클래스와 AP 클래스는 이전 코드에서 정의된 것을 사용합니다.
        multilat = Multilateration(aps)
        estimated_location = multilat.calcUserLocation()
        
    print(f"실제 위치: {uav_state[:2]}")
    print(f"추정 위치: {estimated_location}")
    
    # 각 AP로부터의 거리 오차를 계산한 후 평균을 구합니다.
    errors = [abs(act - est) for act, est in zip(uav_state[:2], estimated_location)]
    average_error = sum(errors) / len(errors)
    print(f"평균 거리 오차: {average_error}")

class AP:
    def __init__(self, x, y, distance):
        self.x = x
        self.y = y
        self.distance = distance

class Multilateration:
    def __init__(self, APs):
        self.APs = APs
    
    def calcUserLocation(self):
        A = []
        b = []
        for i in range(len(self.APs)):
            if i < len(self.APs) - 1:
                x_diff = 2 * (self.APs[i+1].x - self.APs[i].x)
                y_diff = 2 * (self.APs[i+1].y - self.APs[i].y)
                dist_diff = self.APs[i].distance**2 - self.APs[i+1].distance**2 \
                            - self.APs[i].x**2 + self.APs[i+1].x**2 \
                            - self.APs[i].y**2 + self.APs[i+1].y**2
                A.append([x_diff, y_diff])
                b.append([dist_diff])
        
        A = np.array(A)
        b = np.array(b)
        
        user_location = np.linalg.lstsq(A, b, rcond=None)[0]
        return user_location.flatten()

# UAV와 AP의 위치 설정
uav_state = [50, 50, 100]  # UAV의 실제 위치 (x, y, z)
ground_state = [0, 0, 0]  # 지상 기준점의 위치

# AP 인스턴스 생성
ap1 = AP(0, 100, 0)
ap2 = AP(100, 0, 0)
ap3 = AP(100, 100, 0)
ap4 = AP(50, 50, 0)  # 추가된 AP

# 함수 실행
printLocationError(uav_state, ground_state, [ap1, ap2, ap3, ap4])

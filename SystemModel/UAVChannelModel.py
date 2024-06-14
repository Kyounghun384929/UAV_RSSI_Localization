import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

frequency = 2e9 # 논문 2GHz 기준
c = 3e8 # 빛 속도 m/s
channel_params = {
    'aLoS': 10, 'bLoS': 2, 'aNLoS': 30, 'bNLoS': 1.7, 'a0' : 47, 'b0' : 20
    }

def cal_distance(r: int, h: int) -> float:
    '''
    params:
    r: Euclidean distance between UAV cell point (x, y) and ground object (x, y)
    h: UAV height
    
    returns:
    Euclidean distance -> float
    '''
    return np.sqrt(r**2 + h**2)

def FSPL(r: int, h:int) -> float:
    '''
    params:
    r: Euclidean distance between UAV cell point (x, y) and ground object (x, y)
    h: UAV height
    
    returns:
    RSSI Free Space Path loss in dB scale
    '''
    d = cal_distance(r, h)
    return -(20 * np.log10(d) + 20 * np.log10(frequency) + 20 * np.log10(4 * np.pi / c))

# def channelmodel(r, h):
#     '''
#     논문에서 제안한 NogNormal Shadowing Model.
#     환경에 대한 사전 지식이 없는 경우 더 유용할 수 있음.
#     하지만 P_Los에 대한 계산은 거리가 가까우면 무조건 Los확률로 계산되기 때문에 적합하지 않음.
#     '''
#     if r == 0:
#         theta = np.pi / 2
#     else:
#         theta = np.arctan(h / r)
    
#     P_Los = 1 / (1 + channel_params['a0'] * np.exp(-channel_params['b0'] * theta))
#     sigma_los = channel_params['aLoS'] * np.exp(-channel_params['bLoS'] * theta)
#     sigma_nlos = channel_params['aNLoS'] * np.exp(-channel_params['bNLoS'] * theta)
    
#     sigma = P_Los**2 * sigma_los**2 + (1 - P_Los)**2 * sigma_nlos**2
    
#     fspl = FSPL(r, h)
#     loss = np.random.normal(0, sigma)
#     return fspl+loss

def channelmodel(r: int, h: int) -> float:
    '''
    params:
    r: Euclidean distance between UAV cell point (x, y) and ground object (x, y)
    h: UAV height
    
    returns:
    RSSI Path loss in dB scale
    '''
    if r == 0:
        theta = np.pi / 2
    else:
        theta = np.arctan(h / r)
        
    P_Los = 1 / (1 + channel_params['a0'] * np.exp(-channel_params['b0'] * theta))
    sigma_los = channel_params['aLoS'] * np.exp(-channel_params['bLoS'] * theta)
    sigma_nlos = channel_params['aNLoS'] * np.exp(-channel_params['bNLoS'] * theta)
    
    sigma = P_Los**2 * sigma_los**2 + (1 - P_Los)**2 * sigma_nlos**2
    
    fspl = FSPL(r, h)
    # loss = np.random.normal(np.sqrt(sigma), np.sqrt(sigma))
    loss = np.random.normal(0, sigma)

    return fspl + loss

def estimate_distance(rssi_measured: float, n: float) -> float:
    '''
    Params:
    rssi_measured: 측정된 path loss dB 신호 세기
    n : Path loss exponent params
    
    Returns:
    estimated_distance: 추정거리 m
    '''
    d0 = 100 # reference distance
    # PL0 = FSPL(r=0, h=100)
    PL0 = channelmodel(r=0, h=100)
    
    d = 10 ** ((PL0-rssi_measured) / (10 * n))
    d *= d0
    
    noise = np.random.normal(0, 0.1 * d)  # 표준편차를 거리의 10%로 설정
    d += noise
    
    return d

if __name__ == "__main__":
    sns.set_style(style='whitegrid')
    d = np.linspace(0, 300, 100)
    
    fig, axs = plt.subplots(2, 1, figsize=(10,10))
    
    cm = [channelmodel(r, 100) for r in d]
    sns.lineplot(x=d, y=FSPL(d, 100), ax=axs[0], c='#000000', label='FSPL')
    sns.lineplot(x=d, y=cm, ax=axs[0], label='Lossy')
    axs[0].set_xlabel('Distances [m]')
    axs[0].set_ylabel('RSSI [dBm]')
    axs[0].legend()
    # 거리에 따른 RSSI path loss 변화량 plot
    # ---------------------------------------------------------------------------- #
    # 거리에 따른 RSSI path loss 값 기반으로 한 거리 추정 plot
    d = np.linspace(0, 100)
    rssi = [channelmodel(r, 100) for r in d]
    md1 = estimate_distance(rssi, n=2.7)
    md2 = estimate_distance(rssi, n=3.5)
    
    sns.lineplot(x=d, y=md1, label='path loss exponent n=2.7', ax=axs[1])
    sns.lineplot(x=d, y=md2, label='path loss exponent n=3.5', ax=axs[1])
    axs[1].set_xlabel('Distances [m]')
    axs[1].set_ylabel('RSSI [dBm]')
    axs[1].legend()
    
    plt.show()
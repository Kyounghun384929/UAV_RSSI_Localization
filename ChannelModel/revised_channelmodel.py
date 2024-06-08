import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def LogNormal_paper(r, h):
    '''
    논문에서 제안한 NogNormal Shadowing Model.
    환경에 대한 사전 지식이 없는 경우 더 유용할 수 있음.
    하지만 P_Los에 대한 계산은 거리가 가까우면 무조건 Los확률로 계산되기 때문에 적합하지 않음.
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
    loss = np.random.normal(0, sigma)
    path_loss = fspl+loss
    return P_Los, path_loss

def LogNormal_proposed_ver1(r, h, a, b, Los=True):
    '''
    논문의 레퍼런스에서 수정함.
    환경에 대한 구체적인 정보가 있는 경우에는 이 방법이 유용할 수 있음.
    '''
    if r == 0:
        theta = np.pi / 2
    else:
        theta = np.arctan(h / r)
        
    sigma = a * np.exp(-b * theta)  # Shadowing effect
    sigma = np.random.normal(sigma, sigma)
    path_loss = FSPL(r, h)
    if Los:
        path_loss += sigma  # Add shadowing for LoS
    else:
        path_loss += sigma  # Add extra loss for NLoS
    return path_loss

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

def estimate_distance(rssi_measured: int, n: float) -> float:
    """
    Args:
        rssi_measured: 측정된 RSSI 값 (dBm)
        n: 경로 손실 지수 parameter
    
    Returns:
        estimated_distance: 추정된 거리 (m)
    """
    ref_distance = 100
    ref_RSSI = -FSPL(r=0, h=100)
    ref_RSSI = LogNormal_proposed_ver2(r=0, h=100)[1]
    
    d = 10 ** ((-ref_RSSI - rssi_measured) / (10 * n))
    d *= ref_distance
    return d

if __name__ == '__main__':
    sns.set_theme(style='whitegrid')
    
    d = np.linspace(0, 300, 100)
    
    # path_loss_FSPL = -FSPL(d, 100)
    # plt.xlabel('Distances [m]')
    # plt.ylabel('RSSI [dBm]')
    # sns.lineplot(x=d, y=path_loss_FSPL, label='FSPL RSSI', c='#000000')
    # plt.show()

    # distance = np.linspace(0, 1000, 1000)
    # P_LOS_Prob = [LogNormal_paper(dis, 100)[0] for dis in distance]
    # sns.lineplot(x=distance, y=P_LOS_Prob)
    # plt.title("Path Loss Probability versus Distances")
    # plt.xlabel('Distances [m]')
    # plt.ylabel('Probability')
    # plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    
    pathloss = [-LogNormal_paper(dis, 100)[1] for dis in d]
    pathloss_Los = [-LogNormal_proposed_ver1(dis, 100, environ_params['aLoS'], environ_params['bLoS']) for dis in d]
    pathloss_NLos = [-LogNormal_proposed_ver1(dis, 100, environ_params['aNLoS'], environ_params['bNLoS']) for dis in d]
    sns.lineplot(x=d, y=pathloss_Los, ax=axs[0], c='orange', label='Log-Normal Shadowing Path Loss in Los', marker='^')
    sns.lineplot(x=d, y=pathloss_NLos, ax=axs[0], label='Log-Normal Shadowing Path Loss in NLoS', marker='s', c='#1F8300')
    sns.lineplot(x=d, y=-FSPL(np.linspace(0, 300, 100), 100), ax=axs[0], c='#000000', label='FSPL')
    axs[0].set_title('Path Loss Comparison')
    axs[0].set_xlabel('Distances [m]')
    axs[0].set_ylabel('RSSI [dBm]')
    axs[0].legend()
    
    sns.lineplot(x=d, y=pathloss, ax=axs[1], c='blue', label='Log-Normal Shadowing Path Loss', marker='o')
    sns.lineplot(x=d, y=-FSPL(np.linspace(0, 300, 100), 100), ax=axs[1], c='#000000', label='FSPL')
    axs[1].set_xlabel('Distances [m]')
    axs[1].set_ylabel('RSSI [dBm]')
    axs[1].legend()
    
    pl = [-LogNormal_proposed_ver2(dis, 100)[1] for dis in d]
    sns.lineplot(x=d, y=pl, ax=axs[2], c='blue', label='Log-Normal Shadowing Path Loss', marker='o')
    sns.lineplot(x=d, y=-FSPL(np.linspace(0, 300, 100), 100), ax=axs[2], c='#000000', label='FSPL')
    axs[2].set_xlabel('Distances [m]')
    axs[2].set_ylabel('RSSI [dBm]')
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    temp = np.linspace(0, 100)
    rssi = [-LogNormal_proposed_ver2(r, 100)[1] for r in temp]
    # print(rssi)
    measured_distance_1 = estimate_distance(rssi, 2)
    measured_distance_2 = estimate_distance(rssi, 4)
    # print(measured_distance_1, measured_distance_2)
    
    sns.lineplot(x=temp, y=measured_distance_1, label='path loss exponent n=2')
    sns.lineplot(x=temp, y=measured_distance_2, label='path loss exponent n=4')
    plt.xlabel('Real Distances [m]')
    plt.ylabel('Distances estimation[m]')
    plt.title('Distance estimation')
    plt.show()

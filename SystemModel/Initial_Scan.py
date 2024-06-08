import numpy as np
from ENV_Cell import *
from UAVChannelModel import *

def initScan(InitScanPoint: list, groundobject: list, D: int, threshold_rssi=-95):
    '''
    Args:
    InitScanPoint: 초기 스캔을 수행할 웨이포인트 리스트
    groundobject: 초기 스캔을 수행할 때 파악할 ground object 리스트
    D: 통신반경 [m]
    
    현재 위치와 ground Object의 간의 Path Loss 값을 통해 
    기준값 (-95 dB) 이내면 개수 저장
    
    Returns:
    Snodes 위치별 Ground Objects 들의 개수
    '''
    data = {}
    for waypoint in InitScanPoint:
        waypoint_pos = np.array(waypoint)
        count = 0
        for g_pos in groundobject:
            ground_pos = np.array([g_pos['x'], g_pos['y']])
            r = np.linalg.norm(waypoint_pos - ground_pos)
            
            if r <= D:
                rssi = channelmodel(r, h=100)
                if threshold_rssi <= rssi <= -80:
                    count += 1
            
        data[tuple(waypoint_pos)] = count
            
    return data

if __name__ == "__main__":
    Lx = 900        # 가로 [m]
    Ly = 700        # 세로 [m]
    h = 100         # 고도 [m]
    D = 200         # 통신반경 [m]
    w = 2           # Cell 나누는 단위
    
    R = np.sqrt(D**2 - h**2)
    Cx = Lx / np.ceil(Lx/R)
    Cy = Ly / np.ceil(Ly/R)
    
    scan_points = Init_ScanWayPoint(Lx, Ly, Cx, Cy)
    gobject = set_gobjects(Lx, Ly)
    data = initScan(scan_points, gobject, D, threshold_rssi=-95)
    print(data)
    
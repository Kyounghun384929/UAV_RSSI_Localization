import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from SystemModel.ENV_Cell import *
from main import GPSDeniedENV

class GroundObjectTracker:
    def __init__(self, Lx, Ly):
        self.Lx = Lx  # x 범위
        self.Ly = Ly  # y 범위
        self.env = None  # 환경 객체 (나중에 설정)

    def generate_ground_objects(self, num_objects):
        """
        지상 객체의 실제 위치를 생성합니다.
        """
        Gobjects = []
        np.random.seed(1230)  # 재현성을 위한 seed 고정
        for i in range(num_objects):
            Gobjects.append({
                'id': i,
                'x': round(np.random.uniform(0, self.Lx)),
                'y': round(np.random.uniform(0, self.Ly)),
            })
        np.random.seed()  # seed 초기화
        return Gobjects

    def generate_estimated_positions(self, ground_objects):
        """
        실제 위치를 기반으로 추정 위치를 생성합니다.
        """
        estimated_pos = []
        for obj in ground_objects:
            error_distance = np.random.uniform(25,65)  # 20~30m 오차
            error_angle = np.random.uniform(0, 2 * np.pi)  # 랜덤 각도
            estimated_pos.append({
                'id': obj['id'],
                'estimated pos': [
                    round(obj['x'] + error_distance * np.cos(error_angle)),
                    round(obj['y'] + error_distance * np.sin(error_angle))
                ]
            })
        return estimated_pos

    def visualize_positions(self):
        """
        실제 위치와 추정 위치를 시각화합니다.
        """
        plt.figure(figsize=(8, 8))
        for obj in self.env.ground_objects:
            plt.scatter(obj['x'], obj['y'], c='r', marker='o', label='Real Position' if obj['id'] == 0 else "")
        for est in self.env.estimated_pos:
            plt.scatter(est['estimated pos'][0], est['estimated pos'][1], c='b', marker='x', label='Estimated Position' if est['id'] == 0 else "")
            for obj in self.env.ground_objects:
                if obj['id'] == est['id']:
                    plt.plot([obj['x'], est['estimated pos'][0]], [obj['y'], est['estimated pos'][1]], c='gray', linestyle='--')

        plt.title(f'Ground Truth vs. Estimated Position')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)

        # 저장 폴더 생성 및 파일 저장
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        foldername = f"{now}"
        folderdir = os.path.join('./Results', foldername)
        os.makedirs(folderdir, exist_ok=True)
        plt.savefig(os.path.join(folderdir, f'plot.png'))
        plt.close()

# 예시 사용 방법 (실제 환경 객체는 별도로 구현해야 합니다.)
tracker = GroundObjectTracker(Lx=900, Ly=700)
env = GPSDeniedENV(900, 700, 100, 200, 2, 4, 30)
tracker.env = env


num_objects = 30
tracker.env.ground_objects = tracker.generate_ground_objects(num_objects)
tracker.env.estimated_pos = tracker.generate_estimated_positions(tracker.env.ground_objects)
tracker.visualize_positions()

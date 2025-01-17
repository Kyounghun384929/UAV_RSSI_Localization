import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
from tqdm import tqdm
from collections import deque
import random
from SystemModel.UAVChannelModel import channelmodel, FSPL
from SystemModel.Localization import multilateration
import os
from datetime import datetime

# 현재 시간을 포맷팅
now = datetime.now().strftime("%Y%m%d_%H_%M")

class GPSDeniedENV:
    def __init__(self, Lx, Ly, h, D, w, n, num_objects):
        self.Lx = Lx
        self.Ly = Ly
        self.h = h
        self.D = D
        self.w = w
        self.n = n  # 경로 감쇄 지수
        self.num_objects = num_objects

        self.R = np.sqrt(D**2 - h**2)
        self.Cx = self.Lx / np.ceil(self.Lx / self.R)
        self.Cy = self.Ly / np.ceil(self.Ly / self.R)

        self.initial_scan_points = self.init_scan_waypoints()
        self.cell_points = self.cell_waypoints()
        self.ground_objects = self.generate_ground_objects(self.num_objects)
        self.initial_scan_target_count = self.init_target_count()

    def get_num_state_action(self):
        """
        Returns: Num_state, Num_actions
        """
        return len(self.cell_points), len(self.cell_points)

    def get_state_index(self, state):
        return self.cell_points.index(state)    

    def init_scan_waypoints(self):
        Snodes = []
        x, y = 0, 0
        
        while x <= self.Lx:
            temp = y
            while y <= self.Ly:
                Snodes.append((x, y))
                y += 2 * self.Cy
            y = self.Cy if temp == 0 else 0
            x += self.Cx
        return Snodes
    
    def cell_waypoints(self):
        Cnodes = []
        x = self.Cx / (2 * self.w)
        y = self.Cy / (2 * self.w)
        
        while x <= self.Lx:
            while y <= self.Ly:
                Cnodes.append((x, y))
                y += self.Cy / self.w
            y = self.Cy / (2 * self.w)
            x += self.Cx / self.w
        return Cnodes
    
    def generate_ground_objects(self, num_objects):
        Gobjects = []
        np.random.seed(1230)
        for i in range(num_objects):
            Gobjects.append({
                'id': i,
                'x': round(np.random.uniform(0, self.Lx)),
                'y': round(np.random.uniform(0, self.Ly)),
                'vx': round(np.random.normal(0, 1.5)),
                'vy': round(np.random.normal(0, 1.5))
            })
        np.random.seed()
        return Gobjects
    
    def init_target_count(self, threshold_rssi=-88):
        data = {}
        for waypoint in self.initial_scan_points:
            waypoint_pos = np.array(waypoint)
            count = 0
            for g_pos in self.ground_objects:
                ground_pos = np.array([g_pos['x'], g_pos['y']])
                r = np.linalg.norm(waypoint_pos - ground_pos)
                
                if r <= self.D:
                    rssi = FSPL(r, self.h)
                    if threshold_rssi <= rssi <= -80:
                        count += 1
            data[tuple(waypoint_pos)] = count
        return data
    
    def reset(self):
        # 위치 초기화
        self.state = self.cell_points[np.random.randint(len(self.cell_points))]
        # 기타 변수 초기화
        self.visited = 0
        self.estimated_pos = []
        self.path_length = 0
        return self.state
    
    def step(self, action):
        # 방문 횟수 Update
        self.visited += 1
        
        next_state = self.cell_points[action]

        # 다음 위치에서 지상 물체에 대해서 RSSI 측정을 진행하고, 지상 물체 'id'에 대한 진행 횟수 저장
        for obj in self.ground_objects:
            r = np.linalg.norm(np.array(next_state) - np.array([obj['x'], obj['y']]))
            if r <= self.D:
                rssi = FSPL(r, self.h)
                obj[f'Aps_{self.visited}'] = (next_state[0], next_state[1], rssi)
                
            # 동일 id로 RSSI 측정 횟수가 3번 이상일 경우, Trilateration 단 1번 실행
            if len(obj) >= 8: # (기본 5개 + 위치&rssi 3개)
                positions = [list(v[:2]) + [v[2]] for k, v in obj.items() if k.startswith('Aps_')]
                x, y = multilateration(positions, n=self.n)
                
                existing_est_pos = next((item for item in self.estimated_pos if item["id"] == obj['id']), None)
                if existing_est_pos:
                    existing_est_pos['estimated pos'] = (x, y)
                else:
                    self.estimated_pos.append({'id': obj['id'], 'estimated pos': (x, y)})
        
        # 이동 경로 길이, 방문 경유 지점 수 등 강화학습 stop parameters 업데이트
        self.path_length += np.linalg.norm(np.array(self.state) - np.array(next_state))
        
        # 현재 위치 업데이트 및 보상 계산 및 에피소드 종료 여부 확인
        self.state = next_state
        reward = self.get_reward(self.state)
        stop = self.is_terminal_state()
        
        return self.state, reward, stop
        
    def calculate_localization_error(self):
        """
        Returns: 모든 Ground Object에 대한 평균 Localization Error
        """
        total_error = 0.
        estimated_count = 0.
        for obj in self.ground_objects:
            if 'estimated pos' in obj:
                estimated_pos = obj['estimated pos']
                error = np.linalg.norm(np.array(estimated_pos) - np.array([obj['x'], obj['y']]))
                total_error += error
                estimated_count += 1
        
        if estimated_count > 0:
            avg_error = total_error / estimated_count
            return avg_error
        else:
            return -(self.visited + self.path_length/100)
        
    def get_reward(self, state):
        total_error = 0
        estimated_count = 0
        for obj in self.ground_objects:
            if f'Aps_{self.visited}' in obj:
                estimated_pos = obj[f'Aps_{self.visited}'][:2]  # (x, y) 좌표만 사용
                error = np.linalg.norm(np.array(estimated_pos) - np.array([obj['x'], obj['y']]))
                estimated_count += 1
                total_error += error
            
        if estimated_count > 0:
            avg_error = total_error / estimated_count
            return -avg_error
        else:
            return -100
        
    def is_terminal_state(self):
        # 1. Cell Way Point 이동 횟수
        # 2. 최대 경로 길이
        return self.visited == 100 or self.path_length >= 7000
    
class QAgent:
    def __init__(self, env=GPSDeniedENV, gamma=0.999, lr=3e-4, buffer_size=10000, batch_size=64):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        
        self.epsilon_start = 1.0
        self.epsilon_end = 1e-4
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        
        self.num_states, self.num_actions = self.env.get_num_state_action()
        self.q_table = np.zeros([self.num_states, self.num_actions])
        
        # 경험 재플레이 버퍼
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def get_epsilon(self, episode):
        # epsilon이 exponential 하게 감소하도록 코드 변경
        if episode % 100 == 0:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
    
    def get_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state_idx])
    
    def learning(self, num_episodes=50000):
        localization_errors = []
        path_lengths = []
        rewards = []
        
        for episode in tqdm(range(num_episodes)):
            state = self.env.reset()
            state_idx = self.env.get_state_index(state)
            self.get_epsilon(episode)
            stop = False
            
            total_reward = 0
            trajectory = [self.env.state]
            
            while not stop:
                action = self.get_action(state_idx)
                next_state, reward, stop = self.env.step(action)
                next_state_idx = self.env.get_state_index(next_state)
                
                # 경험 재플레이 버퍼에 현재 transition 저장
                self.replay_buffer.append((state_idx, action, reward, next_state_idx, stop))
                
                state_idx = next_state_idx
                trajectory.append(next_state)
                total_reward += reward
                
                if stop:
                    if episode+1 == 1 or episode % 1000 == 0:
                        self.visualize_positions(episode, trajectory)
                    
                    localization_errors.append(self.env.calculate_localization_error())
                    rewards.append(total_reward)
                    path_lengths.append(self.env.path_length)
                    self.env.path_length = 0
                    break
            
            # Q-테이블 업데이트
            self.update_q_table()
        
        self.visualize_results(localization_errors, path_lengths, rewards, num_episodes)
        self.save_q_table()
        
        return localization_errors, path_lengths, rewards
    
    def update_q_table(self):
        """
        경험 재플레이를 통해 Q-테이블을 업데이트 합니다.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        for state_idx, action, reward, next_state_idx, stop in batch:
            q_predict = self.q_table[state_idx, action]
            if stop:
                q_target = reward
            else:
                q_target = reward + self.gamma * np.max(self.q_table[next_state_idx])
            
            self.q_table[state_idx, action] += self.lr * (q_target - q_predict)
    
    def visualize_positions(self, episode, trajectory):
        plt.figure(figsize=(8, 8))
        for obj in self.env.ground_objects:
            plt.scatter(obj['x'], obj['y'], c='r', marker='o', label='Real Position' if obj['id'] == 0 else "")  # Ground Truth (빨간색)
        
        for i, est in enumerate(self.env.estimated_pos):
            if i == 0:
                plt.scatter(est['estimated pos'][0], est['estimated pos'][1], c='b', marker='x', label='Estimated Position')
            else:
                plt.scatter(est['estimated pos'][0], est['estimated pos'][1], c='b', marker='x')
        
            for obj in self.env.ground_objects:
                if obj['id'] == est['id']:
                    plt.plot([obj['x'], est['estimated pos'][0]], [obj['y'], est['estimated pos'][1]], c='gray', linestyle='--')
        
        plt.title(f'Ground Truth vs. Estimated Position (Episode {episode + 1})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        foldername = f"{now}_gamma_{self.gamma}_lr_{self.lr}"
        folderdir = os.path.join('./Results', foldername)
        os.makedirs(folderdir, exist_ok=True)
        plt.savefig(os.path.join(folderdir, f'episode_{episode+1}_plot.png'))  # 각 에피소드별로 그림 저장
        plt.close()  # 메모리 절약을 위해 그림 닫기

    def visualize_results(self, localization_errors, path_lengths, rewards, num_episodes):
        # 에피소드별 Localization Error 시각화
        chart1 = alt.Chart(pd.DataFrame({'episode': range(num_episodes), 'localization_error': localization_errors})).mark_line().encode(
            x='episode',
            y='localization_error',
            tooltip=['episode', 'localization_error']
        ).properties(title='Localization Error per Episode').interactive()
        chart1.save('./Out/localization_error_per_episode.json')

        # 에피소드별 Path Length 시각화
        chart2 = alt.Chart(pd.DataFrame({'episode': range(num_episodes), 'path_length': path_lengths})).mark_line().encode(
            x='episode',
            y='path_length',
            tooltip=['episode', 'path_length']
        ).properties(title='Path Length per Episode').interactive()
        chart2.save('./Out/path_length_per_episode.json')

        # 에피소드별 Reward 시각화
        chart3 = alt.Chart(pd.DataFrame({'episode': range(num_episodes), 'reward': rewards})).mark_line().encode(
            x='episode',
            y='reward',
            tooltip=['episode', 'reward']
        ).properties(title='Reward per Episode').interactive()
        chart3.save('./Out/reward_per_episode.json')

    def save_q_table(self, filename="./Out/q_table_multilateration.json"):
        """
        Q-table을 JSON 파일로 저장합니다.
        """
        import json
        with open(filename, "w") as f:
            json.dump(self.q_table.tolist(), f)

if __name__ == "__main__":
    env = GPSDeniedENV(900, 700, 100, 200, 2, 4, 30)
    agent = QAgent(env, gamma=0.999)
    localization_errors, path_lengths, rewards = agent.learning(num_episodes=50000)

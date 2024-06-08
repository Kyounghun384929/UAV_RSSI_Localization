import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
import altair as alt
import pandas as pd

def visualize_q_table(filename="q_table.json"):
    """
    JSON 파일에서 Q-table을 불러와 히트맵으로 시각화합니다.
    """
    with open(filename, "r") as f:
        q_table = np.array(json.load(f))

    plt.figure(figsize=(10, 8))
    sns.heatmap(q_table, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.title("Q-table Heatmap")
    plt.show()

def visualize_json(filename, title):
    """JSON 파일을 읽어와 Altair 차트를 생성하고 표시합니다."""
    with open(filename, 'r') as f:
        data = json.load(f)

    key = list(data['datasets'].keys())[0]
    df = pd.DataFrame(data['datasets'][key])
    df.columns = ['episode', title.lower()]  # 컬럼명 설정

    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('episode:Q', title='Episode'),
        y=alt.Y(f'{title.lower()}:Q', title=title),
        tooltip=['episode', title.lower()]
    ).properties(title=title).interactive()

    return chart

if __name__ == "__main__":
    # JSON 파일 경로 설정 (실제 경로로 변경해야 합니다)
    localization_error_file = 'localization_error_per_episode.json'
    path_length_file = 'path_length_per_episode.json'
    reward_file = 'reward_per_episode.json'
        
    # 차트 생성 및 표시
    chart1 = visualize_json(localization_error_file, 'Localization Error')
    chart2 = visualize_json(path_length_file, 'Path Length')
    chart3 = visualize_json(reward_file, 'Reward')

    chart1.show()
    chart2.show()
    chart3.show()
    
    # visualize_q_table()

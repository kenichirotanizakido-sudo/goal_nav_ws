"""AI（Q学習）の「頭脳」の部分.

学習（train.py）とロボットの制御（goal_nav_node.py）の両方がこのファイルを使う.

状態: LiDAR で見た障害物（左・正面・右の 3 つの範囲に障害物があるか）× ゴールの方向（5 通り）
      2 x 2 x 2 x 5 = 40 通り
行動: 前進・左へ曲がる・右へ曲がる の 3 つ
Q テーブル: 40 行 x 3 列の表. 「この状態でこの行動をとると、どれくらい良いか」の点数が入っている.
            ロボットは、いまの状態の行で一番点数が高い行動を選ぶ.
"""
import csv
import math

import numpy as np

# 行動: (名前, 前に進む速さ [m/s], 回る速さ [rad/s])  ※ 左回りがプラス
# 曲がるときも少しだけ前に進む（その場で回るだけだと、左右に向きを変え続けて止まってしまうことがあるため）
ACTIONS = [
    ('前進', 0.22, 0.0),
    ('左へ曲がる', 0.06, 0.8),
    ('右へ曲がる', 0.06, -0.8),
]

# 障害物を調べる 3 つの範囲（ロボットから見て、前が x、左が y [m]）
#   正面: ロボットが前に進むと通る道（幅 ±0.3 m）の、前 0.75 m まで
#   左・右: その道のすぐ外側（横 0.3〜0.7 m、前後 -0.3〜0.75 m）
FRONT_LEN = 0.75
HALF_WIDTH = 0.30   # ロボットの半径 0.2 m + 余裕 0.1 m
SIDE_WIDTH = 0.40
GOAL_DIRS = ['左うしろ', '左', '正面', '右', '右うしろ']
N_STATES = 2 * 2 * 2 * len(GOAL_DIRS)


def lidar_sectors(angles, ranges):
    """LiDAR の点を (前 x, 左 y) の位置に直し、左・正面・右の範囲に障害物があるか（True/False）を返す."""
    ok = np.isfinite(ranges)  # 何も当たらなかった点（inf）は使わない
    x = ranges[ok] * np.cos(angles[ok])
    y = ranges[ok] * np.sin(angles[ok])
    near_x = (x > -0.3) & (x < FRONT_LEN)
    front = bool(np.any((x > 0) & (x < FRONT_LEN) & (np.abs(y) <= HALF_WIDTH)))
    left = bool(np.any(near_x & (y > HALF_WIDTH) & (y <= HALF_WIDTH + SIDE_WIDTH)))
    right = bool(np.any(near_x & (y < -HALF_WIDTH) & (y >= -HALF_WIDTH - SIDE_WIDTH)))
    return left, front, right


def goal_direction(angle):
    """ゴールの方向（ロボットの正面から何度ずれているか、左がプラス）を 5 通りに分ける."""
    d = math.degrees(angle)
    if d > 60:
        return 0  # 左うしろ
    if d > 15:
        return 1  # 左
    if d >= -15:
        return 2  # 正面
    if d >= -60:
        return 3  # 右
    return 4      # 右うしろ


def get_state(left, front, right, goal_angle):
    """障害物の有無（True/False）とゴールの方向から、状態の番号（0〜39）を作る."""
    obstacle = int(left) * 4 + int(front) * 2 + int(right)
    return obstacle * len(GOAL_DIRS) + goal_direction(goal_angle)


def describe(state):
    """状態の番号を、人が読める形（左・正面・右の障害物、ゴールの方向）に戻す."""
    obstacle, goal = divmod(state, len(GOAL_DIRS))
    left, front, right = (obstacle >> 2) & 1, (obstacle >> 1) & 1, obstacle & 1
    yn = ['なし', 'あり']
    return yn[left], yn[front], yn[right], GOAL_DIRS[goal]


def choose_action(q_table, state):
    """Q テーブルの、いまの状態の行で一番点数が高い行動を選ぶ."""
    return int(np.argmax(q_table[state]))


def save_q_table(path, q_table):
    """Q テーブルを CSV で保存する（Excel などでも開ける）."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['状態', '左の障害物', '正面の障害物', '右の障害物', 'ゴールの方向']
                   + [a[0] for a in ACTIONS] + ['選ぶ行動'])
        for s in range(N_STATES):
            w.writerow([s, *describe(s)] + [f'{v:.4f}' for v in q_table[s]]
                       + [ACTIONS[choose_action(q_table, s)][0]])


def load_q_table(path):
    """save_q_table で保存した CSV を読み込む."""
    with open(path, encoding='utf-8') as f:
        rows = list(csv.reader(f))[1:]
    return np.array([[float(v) for v in row[5:5 + len(ACTIONS)]] for row in rows])

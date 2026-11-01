"""Q 学習の練習用の、とても簡単な 2D シミュレータ（Gazebo は使わない）.

Gazebo で 1 回走らせると 1 分ほどかかるが、ここなら一瞬で終わるので、何千回も試行錯誤させて学習できる.
ロボットは半径 0.2 m の円、障害物は円と長方形で表す. LiDAR は「光線が最初に当たる距離」を計算してまねる.
"""
import math

import numpy as np

from goal_nav.q_agent import ACTIONS, get_state, lidar_sectors

ROBOT_R = 0.20   # ロボットの半径 [m]（Gazebo のロボットと同じ）
SAFETY_MARGIN = 0.05  # 学習では、障害物にこれだけ余裕をもって近づいた時点で「ぶつかった」とする [m]
GOAL_R = 0.30    # ゴールの中心からこの距離に入ったら到着 [m]
DT = 0.1         # 1 ステップの時間 [s]（LiDAR の更新 10 Hz に合わせる）
RAY_ANGLES = np.radians(np.arange(-135, 136, 3))  # LiDAR の光線の向き（3 度おき）
MAX_RANGE = 8.0

# Gazebo のワールド（models/worlds/goal_world.sdf）と同じ配置
# 障害物: ('circle', 中心x, 中心y, 半径) / ('box', 中心x, 中心y, x方向の半分の長さ, y方向の半分の長さ)
GAZEBO_LAYOUT = (
    (0.0, 0.0, 0.0),   # スタート (x, y, 向き)
    (10.0, 0.0),       # ゴール (x, y)
    [('box', 3.3, -0.2, 0.25, 1.2),    # 壁 1
     ('box', 5.8, 1.0, 0.25, 1.2),     # 壁 2
     ('circle', 8.3, -0.4, 0.35)],     # 柱
)


def surface_distance(ob, x, y):
    """点 (x, y) から障害物の表面までの距離."""
    if ob[0] == 'circle':
        _, cx, cy, r = ob
        return math.hypot(x - cx, y - cy) - r
    _, cx, cy, hx, hy = ob
    return math.hypot(max(abs(x - cx) - hx, 0.0), max(abs(y - cy) - hy, 0.0))


def bounding_radius(ob):
    return ob[3] if ob[0] == 'circle' else math.hypot(ob[3], ob[4])


class Sim2D:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, layout=None):
        """配置を決めて最初の状態を返す. layout が None ならランダムな配置を作る."""
        if layout is None:
            layout = self.random_layout()
        (self.x, self.y, self.yaw), self.goal, self.obstacles = layout
        self.best_dist = self.goal_distance()
        return self.state()

    def random_layout(self):
        """スタートとゴールの間に 1〜4 個の障害物を置いた、ランダムな配置を作る."""
        rng = self.rng
        length = rng.uniform(5.0, 10.0)
        phi = rng.uniform(-math.pi, math.pi)
        goal = (length * math.cos(phi), length * math.sin(phi))
        start = (0.0, 0.0, phi + rng.uniform(-math.pi / 2, math.pi / 2))
        obstacles = []
        for _ in range(rng.integers(1, 5)):
            for _ in range(20):
                t = rng.uniform(0.2, 0.8) * length
                side = rng.uniform(-1.0, 1.0)
                cx = t * math.cos(phi) - side * math.sin(phi)
                cy = t * math.sin(phi) + side * math.cos(phi)
                if rng.random() < 0.5:
                    ob = ('circle', cx, cy, rng.uniform(0.2, 0.6))
                else:
                    ob = ('box', cx, cy, rng.uniform(0.15, 1.2), rng.uniform(0.15, 1.2))
                # スタート・ゴールの近くには置かない. 障害物どうしの間はロボットが通れるだけあける
                ok = (surface_distance(ob, 0, 0) > 0.8 and surface_distance(ob, *goal) > 0.8
                      and all(math.hypot(cx - o[1], cy - o[2]) > bounding_radius(ob) + bounding_radius(o) + 0.9
                              for o in obstacles))
                if ok:
                    obstacles.append(ob)
                    break
        return start, goal, obstacles

    def scan(self):
        """LiDAR のまねをして、各方向の障害物までの距離を求める."""
        a = self.yaw + RAY_ANGLES
        dx, dy = np.cos(a), np.sin(a)
        dist = np.full(len(a), MAX_RANGE)
        for ob in self.obstacles:
            if ob[0] == 'circle':
                _, cx, cy, r = ob
                ox, oy = self.x - cx, self.y - cy
                b = dx * ox + dy * oy
                disc = b * b - (ox * ox + oy * oy - r * r)
                t = -b - np.sqrt(np.maximum(disc, 0.0))
                hit = (disc >= 0) & (t > 0)
            else:
                _, cx, cy, hx, hy = ob
                with np.errstate(divide='ignore', invalid='ignore'):
                    tx1, tx2 = (cx - hx - self.x) / dx, (cx + hx - self.x) / dx
                    ty1, ty2 = (cy - hy - self.y) / dy, (cy + hy - self.y) / dy
                t = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
                hit = (np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2)) >= t) & (t > 0)
            dist = np.where(hit & (t < dist), t, dist)
        return dist

    def goal_distance(self):
        return math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)

    def goal_angle(self):
        """ゴールの方向（ロボットの正面から何ラジアンずれているか、左がプラス）."""
        a = math.atan2(self.goal[1] - self.y, self.goal[0] - self.x) - self.yaw
        return math.atan2(math.sin(a), math.cos(a))

    def state(self):
        left, self.front_blocked, right = lidar_sectors(RAY_ANGLES, self.scan())
        return get_state(left, self.front_blocked, right, self.goal_angle())

    def step(self, action):
        """行動を 1 ステップ（0.1 秒）実行して、(次の状態, 報酬, 終わったか, 結果) を返す."""
        _, v, w = ACTIONS[action]
        forward_into_obstacle = action == 0 and self.front_blocked  # 正面に障害物があるのに前進した
        self.yaw += w * DT
        self.x += v * math.cos(self.yaw) * DT
        self.y += v * math.sin(self.yaw) * DT
        dist = self.goal_distance()
        # 報酬: ゴールへの「いちばん近づいた記録」を更新した分だけプラス. 時間がかかると少しずつマイナス
        # （近づいた・遠ざかったで毎回点をつけると、障害物をよけるための回り道をしなくなるため）
        reward = 5.0 * max(0.0, self.best_dist - dist) - 0.02
        if forward_into_obstacle:
            reward -= 1.0
        self.best_dist = min(self.best_dist, dist)
        # 障害物の表面まで 0.25 m（ロボットの半径 + 余裕 5 cm）より近づいたら「ぶつかった」とする
        if any(surface_distance(ob, self.x, self.y) < ROBOT_R + SAFETY_MARGIN for ob in self.obstacles):
            return self.state(), -20.0, True, 'collision'  # ぶつかったら大きなマイナス
        if dist < GOAL_R:
            return self.state(), 20.0, True, 'goal'        # ゴールに着いたら大きなプラス
        return self.state(), reward, False, ''

"""Q 学習で「障害物をよけてゴールへ行く方法」を AI に覚えさせる.

使い方（src/goal_nav フォルダで実行）:
    python3 -m goal_nav.train
できるもの:
    config/q_table.csv        学習した Q テーブル（ロボットはこれを読んで動く）
    docs/learning_curve.png   学習が進むにつれて成功率が上がっていく様子
    docs/sim_trajectory.png   学習後の AI に、Gazebo と同じ配置を 2D シミュレータで走らせた軌跡
"""
import os

import numpy as np

from goal_nav.q_agent import ACTIONS, N_STATES, save_q_table
from goal_nav.sim2d import GAZEBO_LAYOUT, Sim2D, bounding_radius

EPISODES = 8000      # 試行の回数（毎回、障害物の配置をランダムに変える）
MAX_STEPS = 1500     # 1 回の試行の最大ステップ数（150 秒）
ALPHA = 0.1          # 学習率: 1 回の経験で点数をどれだけ書き換えるか
GAMMA = 0.98         # 割引率: 先にもらえる報酬をどれだけ重く見るか
EPS_START, EPS_END = 1.0, 0.05  # ε-greedy: ランダムな行動を試す割合（はじめは多く、だんだん減らす）
SEED = 1             # 乱数の種（0, 1, 2 を試して、テストの成功率が一番高かった 1 を使う）


def run_episode(sim, q, eps=0.0, layout=None, learn=False):
    """1 回の試行（スタートからゴール・衝突・時間切れまで）. 結果と軌跡を返す."""
    s = sim.reset(layout)
    path = [(sim.x, sim.y)]
    for _ in range(MAX_STEPS):
        if sim.rng.random() < eps:
            a = int(sim.rng.integers(len(ACTIONS)))  # たまにランダムな行動を試す
        else:
            a = int(np.argmax(q[s]))                 # いつもは一番点数が高い行動
        s2, r, done, result = sim.step(a)
        if learn:
            # Q 学習の更新式: Q(s,a) ← Q(s,a) + α × (報酬 + γ × 次の状態の一番高い点数 − Q(s,a))
            target = r if done else r + GAMMA * q[s2].max()
            q[s, a] += ALPHA * (target - q[s, a])
        s = s2
        path.append((sim.x, sim.y))
        if done:
            return result, path
    return 'timeout', path


def train(episodes=EPISODES, seed=SEED):
    sim = Sim2D(seed)
    q = np.zeros((N_STATES, len(ACTIONS)))  # はじめは何も知らない（全部 0 点）
    results = []
    for ep in range(episodes):
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * ep / (0.7 * episodes))
        result, _ = run_episode(sim, q, eps=eps, learn=True)
        results.append(result)
        if (ep + 1) % 500 == 0:
            recent = results[-500:]
            print(f'{ep + 1:5d} 回目まで: 成功 {recent.count("goal") / 5:5.1f}%  '
                  f'衝突 {recent.count("collision") / 5:5.1f}%  時間切れ {recent.count("timeout") / 5:5.1f}%  '
                  f'(ε = {eps:.2f})')
    return q, results


def evaluate(q, n=1000, seed=12345):
    """学習に使っていない配置を n 個作って、ランダムな行動なしで試す."""
    sim = Sim2D(seed)
    results = [run_episode(sim, q)[0] for _ in range(n)]
    return {k: results.count(k) / n for k in ('goal', 'collision', 'timeout')}


def plot(results, q, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']

    # 学習曲線: 直近 200 回の成功率
    ok = np.array([r == 'goal' for r in results], dtype=float)
    rate = np.convolve(ok, np.ones(200) / 200, mode='valid') * 100
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    ax.plot(np.arange(200, len(ok) + 1), rate, color='#1f6fd1')
    ax.set_xlabel('試行の回数')
    ax.set_ylabel('成功率 [%]（直近 200 回）')
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'learning_curve.png'))
    plt.close(fig)

    # Gazebo と同じ配置での軌跡
    result, path = run_episode(Sim2D(0), q, layout=GAZEBO_LAYOUT)
    (sx, sy, _), (gx, gy), obstacles = GAZEBO_LAYOUT
    fig, ax = plt.subplots(figsize=(6, 3.0), dpi=150)
    for ob in obstacles:
        if ob[0] == 'circle':
            ax.add_patch(Circle((ob[1], ob[2]), ob[3], color='#7a7f87'))
        else:
            ax.add_patch(Rectangle((ob[1] - ob[3], ob[2] - ob[4]), 2 * ob[3], 2 * ob[4], color='#7a7f87'))
    p = np.array(path)
    ax.plot(p[:, 0], p[:, 1], color='#e8730c', lw=2, label='ロボットの軌跡')
    ax.plot(sx, sy, 'o', color='#2ca02c', ms=12, label='スタート')
    ax.plot(gx, gy, '*', color='#d62728', ms=16, label='ゴール')
    ax.set_aspect('equal')
    ys = [y for o in obstacles for y in (o[2] - bounding_radius(o), o[2] + bounding_radius(o))]
    ax.set_xlim(min(sx, gx) - 1, max(sx, gx) + 1)
    ax.set_ylim(min(*p[:, 1], *ys) - 0.5, max(*p[:, 1], *ys) + 0.5)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(f'2D シミュレータでの確認（結果: {result}, {len(path) - 1} ステップ）', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'sim_trajectory.png'))
    plt.close(fig)


def main():
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    q, results = train()
    stats = evaluate(q)
    print(f'学習後のテスト（学習に使っていない 1000 配置）: 成功 {stats["goal"]:.1%}  衝突 {stats["collision"]:.1%}  '
          f'時間切れ {stats["timeout"]:.1%}')
    os.makedirs(os.path.join(pkg_dir, 'config'), exist_ok=True)
    os.makedirs(os.path.join(pkg_dir, 'docs'), exist_ok=True)
    save_q_table(os.path.join(pkg_dir, 'config', 'q_table.csv'), q)
    plot(results, q, os.path.join(pkg_dir, 'docs'))
    print('保存しました: config/q_table.csv, docs/learning_curve.png, docs/sim_trajectory.png')


if __name__ == '__main__':
    main()

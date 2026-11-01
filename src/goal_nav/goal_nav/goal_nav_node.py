"""Q 学習で学んだ AI を使って、ロボットをスタートからゴールまで動かす ROS 2 ノード.

参考: 2DRoboPrac_ws の src/lidar_node.cpp（LiDAR を見て /cmd_vel を出すノード）
受け取るトピック: /scan（LiDAR）, /odom（ロボットの位置と向き）
出すトピック:     /cmd_vel（速度の指令）
                  /ai_status（AI の判断を文字で。動画に表示する）, /goal_reached（ゴールに着いたら True）
"""
import math
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from goal_nav.q_agent import ACTIONS, choose_action, describe, get_state, lidar_sectors, load_q_table

GOAL_R = 0.30  # ゴールの中心からこの距離に入ったら到着 [m]（学習のときと同じ）


class GoalNavNode(Node):
    def __init__(self):
        super().__init__('goal_nav_node')
        default_q = os.path.join(get_package_share_directory('goal_nav'), 'config', 'q_table.csv')
        self.goal_x = self.declare_parameter('goal_x', 10.0).value
        self.goal_y = self.declare_parameter('goal_y', 0.0).value
        self.start_delay = self.declare_parameter('start_delay', 3.0).value  # 動き出すまで待つ時間 [s]
        self.q_table = load_q_table(self.declare_parameter('q_table', default_q).value)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/ai_status', 10)
        self.goal_pub = self.create_publisher(Bool, '/goal_reached', 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, 10)

        self.pose = None          # ロボットの位置と向き (x, y, yaw)
        self.first_scan = None    # 最初に LiDAR のデータが来た時刻
        self.start_time = None    # 動き出した時刻
        self.reached = False
        self.distance = 0.0       # 走った距離 [m]
        self.closest = 100.0      # 障害物に一番近づいたときの距離（LiDAR から）[m]
        self.counts = [0] * len(ACTIONS)
        self.get_logger().info(f'ゴール ({self.goal_x}, {self.goal_y}) を目指します')

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        if self.pose is not None and self.start_time is not None and not self.reached:
            self.distance += math.hypot(p.x - self.pose[0], p.y - self.pose[1])
        self.pose = (p.x, p.y, yaw)

    def on_scan(self, msg):
        """LiDAR のデータが来るたび（0.1 秒ごと）に、AI に次の行動を決めさせる."""
        if self.pose is None or self.reached:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if self.first_scan is None:
            self.first_scan = now
        if now - self.first_scan < self.start_delay:
            self.publish_status('スタート地点で待っています')
            return
        if self.start_time is None:
            self.start_time = now
            self.get_logger().info('出発します')

        x, y, yaw = self.pose
        dist = math.hypot(self.goal_x - x, self.goal_y - y)
        if dist < GOAL_R:
            self.arrive(now)
            return

        # 1. いまの状態を調べる（左・正面・右に障害物があるか、ゴールはどちらの方向か）
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        left, front, right = lidar_sectors(angles, ranges)
        goal_angle = math.atan2(self.goal_y - y, self.goal_x - x) - yaw
        goal_angle = math.atan2(math.sin(goal_angle), math.cos(goal_angle))  # -180〜180 度に直す
        state = get_state(left, front, right, goal_angle)

        # 2. Q テーブルで一番点数が高い行動を選び、速度の指令を出す
        action = choose_action(self.q_table, state)
        name, v, w = ACTIONS[action]
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        self.counts[action] += 1
        valid = ranges[np.isfinite(ranges)]
        if len(valid):
            self.closest = min(self.closest, float(valid.min()))
        l, f, r, g = describe(state)
        self.publish_status(f'AI（Q学習）の判断: {name}\n'
                            f'障害物  左:{l}  正面:{f}  右:{r}   ゴールの方向: {g}\n'
                            f'ゴールまで {dist:.1f} m   出発から {now - self.start_time:.1f} 秒')

    def arrive(self, now):
        """ゴールに着いたら止まって、結果を表示する."""
        self.reached = True
        self.cmd_pub.publish(Twist())  # 速度 0 = 止まる
        elapsed = now - self.start_time
        self.publish_status(f'ゴールに到着しました！\n所要時間 {elapsed:.1f} 秒 / 走った距離 {self.distance:.1f} m')
        self.goal_pub.publish(Bool(data=True))
        counts = ', '.join(f'{a[0]} {c} 回' for a, c in zip(ACTIONS, self.counts))
        self.get_logger().info(f'ゴールに到着: {elapsed:.1f} 秒, {self.distance:.2f} m 走行, '
                               f'障害物に一番近づいた距離 {self.closest:.2f} m（LiDAR から）, 行動: {counts}')

    def publish_status(self, text):
        self.status_pub.publish(String(data=text))


def main():
    rclpy.init()
    node = GoalNavNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()

"""デモ動画を作るためのノード（ロボットの制御には関係ない）.

Gazebo の撮影用カメラの映像（/video_camera/image）に、AI の判断（/ai_status）を文字で重ねて MP4 にする.
シミュレーションの時刻に合わせてコマを書き込むので、動画は等倍速になる.
ゴールに着いたら（/goal_reached）、数秒後に録画を終えてノードを終了する.
"""
import os
import subprocess

import rclpy
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'  # 日本語フォント（fonts-noto-cjk）
FPS = 30


class VideoRecorder(Node):
    def __init__(self):
        super().__init__('video_recorder')
        self.output = os.path.expanduser(self.declare_parameter('output', '~/goal_nav_video.mp4').value)
        self.end_after = self.declare_parameter('end_after', 4.0).value  # ゴール後も録画を続ける秒数
        self.font = ImageFont.truetype(FONT, 24)
        self.small_font = ImageFont.truetype(FONT, 20)
        self.status = ''
        self.ffmpeg = None
        self.t0 = None         # 最初のコマの時刻
        self.t = 0.0           # 最新のコマの時刻（録画開始からの秒数）
        self.frames = 0
        self.goal_time = None
        self.done = False
        self.create_subscription(Image, '/video_camera/image', self.on_image, 10)
        self.create_subscription(String, '/ai_status', self.on_status, 10)
        self.create_subscription(Bool, '/goal_reached', self.on_goal, 10)

    def on_status(self, msg):
        self.status = msg.data

    def on_goal(self, msg):
        if msg.data and self.goal_time is None:
            self.goal_time = self.t

    def on_image(self, msg):
        if self.done:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        if self.ffmpeg is None:
            self.t0 = stamp
            self.ffmpeg = subprocess.Popen(
                ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                 '-s', f'{msg.width}x{msg.height}', '-r', str(FPS), '-i', '-',
                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', self.output],
                stdin=subprocess.PIPE)
            self.get_logger().info(f'録画を始めました: {self.output}')
        self.t = stamp - self.t0

        img = PILImage.frombytes('RGB', (msg.width, msg.height), bytes(msg.data))
        self.draw_text(img)
        data = img.tobytes()
        while self.frames <= self.t * FPS:  # コマが抜けても、時刻に合わせて同じコマを書き足す
            self.ffmpeg.stdin.write(data)
            self.frames += 1

        if self.goal_time is not None and self.t - self.goal_time > self.end_after:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()
            self.get_logger().info(f'録画を終えました: {self.frames / FPS:.1f} 秒')
            self.done = True

    def draw_text(self, img):
        """左上に AI の判断、左下に凡例を書く（半透明の黒い板の上に白い文字）."""
        overlay = PILImage.new('RGBA', img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        lines = self.status.split('\n') if self.status else ['準備中']
        width = max(d.textlength(line, font=self.font) for line in lines)
        d.rounded_rectangle((16, 16, 44 + width, 26 + 34 * len(lines)), radius=10, fill=(0, 0, 0, 150))
        for i, line in enumerate(lines):
            d.text((30, 22 + 34 * i), line, font=self.font, fill=(255, 255, 255, 255))
        h = img.size[1]
        d.rounded_rectangle((16, h - 56, 600, h - 16), radius=10, fill=(0, 0, 0, 150))
        x = 30
        for color, label in [((40, 180, 60), 'スタート'), ((220, 30, 30), 'ゴール'),
                             ((77, 92, 115), '障害物'), ((242, 128, 26), 'ロボット')]:
            d.ellipse((x, h - 44, x + 16, h - 28), fill=color + (255,))
            d.text((x + 22, h - 50), label, font=self.small_font, fill=(255, 255, 255, 255))
            x += 130
        img.paste(PILImage.alpha_composite(img.convert('RGBA'), overlay).convert('RGB'))


def main():
    rclpy.init()
    node = VideoRecorder()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    except (KeyboardInterrupt, ExternalShutdownException):
        if node.ffmpeg is not None:
            node.ffmpeg.stdin.close()
            node.ffmpeg.wait()


if __name__ == '__main__':
    main()

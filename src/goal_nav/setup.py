from glob import glob

from setuptools import setup

package_name = 'goal_nav'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/models/lidar_bot', glob('models/lidar_bot/*')),
        ('share/' + package_name + '/models/worlds', glob('models/worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='goal_nav',
    maintainer_email='goal_nav@example.com',
    description='Q 学習（強化学習）で障害物をよけてゴールへ向かう LiDAR ロボット（ROS 2 + Gazebo）',
    license='MIT',
    entry_points={
        'console_scripts': [
            'goal_nav_node = goal_nav.goal_nav_node:main',
            'video_recorder = goal_nav.video_recorder:main',
            'train = goal_nav.train:main',
        ],
    },
)

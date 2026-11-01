"""Gazebo・ロボット・ブリッジ・AI のノードをまとめて起動する.

参考: 2DRoboPrac_ws の launch/nav_spawn.launch.py（Gazebo の起動・ロボットのスポーン・ブリッジ）

使い方:
  ros2 launch goal_nav goal_nav.launch.py                           # Gazebo の画面を出して実行
  ros2 launch goal_nav goal_nav.launch.py gui:=false record:=true   # 画面なしで実行して動画を作る
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, RegisterEventHandler
from launch.actions import SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share_dir = get_package_share_directory('goal_nav')
    world = os.path.join(pkg_share_dir, 'models', 'worlds', 'goal_world.sdf')
    robot = os.path.join(pkg_share_dir, 'models', 'lidar_bot', 'model.sdf')
    gz_sim_launch = os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
    gui = LaunchConfiguration('gui')
    use_sim_time = {'use_sim_time': True}

    actions = [
        DeclareLaunchArgument('gui', default_value='true', description='Gazebo の画面を出すか'),
        DeclareLaunchArgument('record', default_value='false', description='動画を録画するか'),
        DeclareLaunchArgument('start_x', default_value='0.0'),
        DeclareLaunchArgument('start_y', default_value='0.0'),
        DeclareLaunchArgument('start_yaw', default_value='0.0'),
        DeclareLaunchArgument('goal_x', default_value='10.0'),
        DeclareLaunchArgument('goal_y', default_value='0.0'),
        DeclareLaunchArgument('video', default_value='~/goal_nav_video.mp4', description='動画の保存先'),
    ]

    # WSL2 で GPU を使って描画するための設定（WSL2 以外では何もしない）
    if os.path.exists('/dev/dxg'):
        actions.append(SetEnvironmentVariable('GALLIUM_DRIVER', 'd3d12'))

    # Gazebo の起動（-r: すぐにシミュレーションを始める。画面なしのときは -s --headless-rendering）
    actions += [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_sim_launch),
            launch_arguments={'gz_args': f'-r {world}', 'on_exit_shutdown': 'true'}.items(),
            condition=IfCondition(gui)),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_sim_launch),
            launch_arguments={'gz_args': f'-r -s --headless-rendering {world}', 'on_exit_shutdown': 'true'}.items(),
            condition=UnlessCondition(gui)),
    ]

    # ロボットをスタート地点にスポーンさせる
    actions.append(Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-name', 'lidar_bot', '-file', robot,
                   '-x', LaunchConfiguration('start_x'), '-y', LaunchConfiguration('start_y'),
                   '-z', '0.0', '-Y', LaunchConfiguration('start_yaw')]))

    # ros_gz_bridge（設定は config/bridge.yaml）
    actions.append(Node(
        package='ros_gz_bridge', executable='parameter_bridge', output='screen',
        parameters=[{'config_file': os.path.join(pkg_share_dir, 'config', 'bridge.yaml')}, use_sim_time]))

    # AI（Q 学習）でロボットを動かすノード
    actions.append(Node(
        package='goal_nav', executable='goal_nav_node', output='screen',
        parameters=[use_sim_time, {'goal_x': ParameterValue(LaunchConfiguration('goal_x'), value_type=float),
                                   'goal_y': ParameterValue(LaunchConfiguration('goal_y'), value_type=float)}]))

    # 動画を作るノード（record:=true のときだけ）。録画が終わったら全体を終了する
    recorder = Node(
        package='goal_nav', executable='video_recorder', output='screen',
        parameters=[use_sim_time, {'output': LaunchConfiguration('video')}],
        condition=IfCondition(LaunchConfiguration('record')))
    actions += [
        recorder,
        RegisterEventHandler(OnProcessExit(target_action=recorder, on_exit=[EmitEvent(event=Shutdown())])),
    ]
    return LaunchDescription(actions)

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='stereo_publisher',
            executable='stereo_publisher_node',
            name='stereo_publisher_node',
            parameters=[{
                'camera_number': 1,
                'device_path': '/dev/video0',
                'calibration_path': PathJoinSubstitution([
                    get_package_share_directory('stereo_publisher'),
                    'calibration'
                ]),
                'verbose_mode': False,
                'save_frames': False,
                'save_location': 'frames/',
            }],
            output='screen'
        )
    ])

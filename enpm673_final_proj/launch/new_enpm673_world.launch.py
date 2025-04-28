from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get path to gazebo_ros package
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # Set path to your world file
    world = os.path.join(
        get_package_share_directory('enpm673_final_proj'),
        'worlds',
        'empty_world.world'  # Change to your world if needed
    )

    print(f"World = {world}")

    # Launch gazebo with the world but WITHOUT GUI
    gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world,
            'gui': 'false'  # <- IMPORTANT: Don't try to launch gzclient inside ROS
        }.items()
    )

    # Create and return launch description
    ld = LaunchDescription()
    ld.add_action(gazebo_cmd)
    return ld

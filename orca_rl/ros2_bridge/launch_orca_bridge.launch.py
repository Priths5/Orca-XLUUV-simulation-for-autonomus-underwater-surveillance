"""
launch_orca_bridge.launch.py
------------------------------
Starts the whole Stonefish + RL stack (no Gazebo, no spawn: the Orca comes
from the Stonefish scenario file loaded by orca_rl_sim):

    ROS 2 launch
      +-- orca_rl_sim            (Stonefish C++ executable, ROS 2 node /orca_stonefish)
      +-- orca_rl_bridge_node.py (Python RL node /orca_rl_bridge)   [mode:=rl only]

    ros2 launch orca_rl/ros2_bridge/launch_orca_bridge.launch.py
    ros2 launch orca_rl/ros2_bridge/launch_orca_bridge.launch.py world:=survey
    ros2 launch orca_rl/ros2_bridge/launch_orca_bridge.launch.py world:=multisub headless:=true rtf:=3
    ros2 launch orca_rl/ros2_bridge/launch_orca_bridge.launch.py mode:=autopilot   # original C++ survey law
    ros2 launch orca_rl/ros2_bridge/launch_orca_bridge.launch.py nav_mode:=DEBUG_GROUND_TRUTH

Arguments: world (rl|survey|multisub), mode (rl|autopilot), headless (bool),
rtf (Stonefish realtime factor), nav_mode (REALISTIC_SENSOR|DEBUG_GROUND_TRUTH),
policy_path, control_rate_hz, sim_build_dir (where orca_rl_sim was built).
Closing the Stonefish window (or Ctrl-C) shuts everything down.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration

HERE = os.path.dirname(os.path.realpath(__file__))
ORCA_RL = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_BUILD = os.path.abspath(os.path.join(ORCA_RL, "..", "stonefish_sim", "build"))
DEFAULT_POLICY = os.path.join(ORCA_RL, "logs", "trained_policy_obstacle.npz")
if not os.path.exists(DEFAULT_POLICY):
    DEFAULT_POLICY = os.path.join(ORCA_RL, "logs", "trained_policy.npz")


def _launch(context):
    g = lambda name: LaunchConfiguration(name).perform(context)
    build = g("sim_build_dir")
    sim_cmd = [os.path.join(build, "orca_rl_sim"), "--world", g("world"), "--mode", g("mode"),
               "--rtf", g("rtf")]
    if g("headless").lower() in ("1", "true", "yes"):
        sim_cmd.append("--headless")
    elif g("follow").lower() in ("1", "true", "yes"):
        sim_cmd.append("--follow")     # camera glued to the Orca, HUD hidden
    sim = ExecuteProcess(cmd=sim_cmd, cwd=build, output="screen", name="orca_rl_sim")
    actions = [sim, RegisterEventHandler(OnProcessExit(target_action=sim,
                                                       on_exit=[EmitEvent(event=Shutdown(reason="orca_rl_sim exited"))]))]
    if g("mode") == "rl":
        actions.append(ExecuteProcess(
            cmd=["python3", os.path.join(HERE, "orca_rl_bridge_node.py"), "--ros-args",
                 "-p", f"world:={g('world')}", "-p", f"nav_mode:={g('nav_mode')}",
                 "-p", f"policy_path:={g('policy_path')}", "-p", f"control_rate_hz:={g('control_rate_hz')}"]
                + (["-p", f"log_dir:={g('log_dir')}"] if g("log_dir") else []),
            cwd=ORCA_RL, output="screen", name="orca_rl_bridge"))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="rl", description="rl | survey | multisub"),
        DeclareLaunchArgument("mode", default_value="rl", description="rl | autopilot"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("follow", default_value="false", description="camera follows the Orca"),
        DeclareLaunchArgument("rtf", default_value="1.0", description="Stonefish realtime factor"),
        DeclareLaunchArgument("nav_mode", default_value="REALISTIC_SENSOR",
                              description="REALISTIC_SENSOR | DEBUG_GROUND_TRUTH"),
        DeclareLaunchArgument("policy_path", default_value=DEFAULT_POLICY),
        DeclareLaunchArgument("control_rate_hz", default_value="1.0"),
        DeclareLaunchArgument("sim_build_dir", default_value=DEFAULT_BUILD),
        DeclareLaunchArgument("log_dir", default_value="", description="RL run logs (default orca_rl/logs/stonefish_runs)"),
        OpaqueFunction(function=_launch),
    ])

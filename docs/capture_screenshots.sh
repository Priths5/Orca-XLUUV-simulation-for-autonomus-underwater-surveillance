#!/usr/bin/env bash
# capture_screenshots.sh - screenshots of the Stonefish window for the slides/report.
# Opens the simulation (X11 needed) with the camera following the Orca while the RL
# policy drives, and saves the window's client area into docs/figures/.
#   docs/capture_screenshots.sh [world] [seconds_before_first_shot] [seconds_between_shots]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$HERE/docs/figures"; mkdir -p "$OUT"
WORLD="${1:-rl}"; WAIT="${2:-20}"; GAP="${3:-40}"
source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-83}"
ros2 launch "$HERE/orca_rl/ros2_bridge/launch_orca_bridge.launch.py" world:="$WORLD" follow:=true log_dir:="$OUT/.capture_runs" \
    > "$OUT/.capture_$WORLD.log" 2>&1 &
LPID=$!
sleep "$WAIT"; python3 "$HERE/docs/_grab_window.py" "$OUT/stonefish_${WORLD}_1.png"
sleep "$GAP";  python3 "$HERE/docs/_grab_window.py" "$OUT/stonefish_${WORLD}_2.png"
kill -INT $LPID; sleep 3; pkill -x orca_rl_sim; wait $LPID 2>/dev/null

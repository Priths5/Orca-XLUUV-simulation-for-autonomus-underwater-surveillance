#!/usr/bin/env bash
# Closed-loop evaluation: Stonefish (headless) + RL node, one run per world.
# Usage: scripts/closed_loop_eval.sh [policy.npz] [rtf] [max_sim_seconds] [worlds...]
# Summaries land in orca_rl/logs/stonefish_runs/*_summary.json
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY="${1:-$HERE/orca_rl/logs/trained_policy_obstacle.npz}"
RTF="${2:-3}"
TMAX="${3:-900}"
shift 3 2>/dev/null || shift $#
WORLDS="${*:-rl survey multisub}"
# (no `set -u`: ROS setup scripts reference unset variables)
source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-47}"
LOGDIR="$HERE/orca_rl/logs/stonefish_runs"
mkdir -p "$LOGDIR"
for W in $WORLDS; do
  echo "=== world $W  (rtf $RTF, max $TMAX sim-s) ==="
  "$HERE/stonefish_sim/build/orca_rl_sim" --world "$W" --headless --rtf "$RTF" > "$LOGDIR/sim_$W.log" 2>&1 &
  SIM=$!
  sleep 3
  timeout $(( TMAX / ${RTF%.*} + 120 )) python3 "$HERE/orca_rl/ros2_bridge/orca_rl_bridge_node.py" --ros-args \
      -p world:="$W" -p policy_path:="$POLICY" -p max_mission_time:="$TMAX" \
      -p exit_on_finish:=true -p log_dir:="$LOGDIR" 2>&1 | grep -E --line-buffered "RUN SUMMARY|CONTACT|waypoint" 
  kill -INT $SIM 2>/dev/null; wait $SIM 2>/dev/null
  grep -c "CONTACT" "$LOGDIR/sim_$W.log" | sed "s/^/contact events in sim log: /"
done

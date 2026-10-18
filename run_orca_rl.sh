#!/usr/bin/env bash
# run_orca_rl.sh - build and run the Orca RL policy inside Stonefish over ROS 2.
#
#   ./run_orca_rl.sh build                 # build stonefish_sim (incl. orca_rl_sim) + run unit tests
#   ./run_orca_rl.sh run [rl|survey|multisub]        # Stonefish window + RL policy (default world: rl)
#   ./run_orca_rl.sh watch [world]         # same, camera follows the Orca (HUD hidden)
#   ./run_orca_rl.sh headless [world] [rtf]          # no window, faster than real time
#   ./run_orca_rl.sh autopilot [world]     # original C++ lawnmower autopilot, no RL
#   ./run_orca_rl.sh legacy [world]        # the original trained_policy.npz instead of the obstacle policy
#   ./run_orca_rl.sh debug [world]         # DEBUG_GROUND_TRUTH navigation (connectivity check only)
#   ./run_orca_rl.sh manual                # sim only; then drive it with orca_rl/tools/manual_cmd.py
#   ./run_orca_rl.sh topics                # print what every Stonefish topic is publishing
#   ./run_orca_rl.sh eval [rtf] [max_s]    # headless closed-loop runs in all 3 worlds + plots
#   ./run_orca_rl.sh train                 # retrain the obstacle-aware policy (~70 min, 14 cores)
#   ./run_orca_rl.sh stop                  # kill any orca_rl_sim left running
#   ./run_orca_rl.sh docs                  # regenerate figures, slide deck and report (docs/)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_BUILD="$HERE/stonefish_sim/build"
RL="$HERE/orca_rl"
LAUNCH="$RL/ros2_bridge/launch_orca_bridge.launch.py"
# ROS setup scripts reference unset variables, so no `set -u` here
source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

need_build() {
    [ -x "$SIM_BUILD/orca_rl_sim" ] || { echo "orca_rl_sim not built - running build first"; build; }
}
build() {
    mkdir -p "$SIM_BUILD" && cd "$SIM_BUILD" && cmake .. && make -j"$(nproc)" && ./orca_rl_adapter_test
    cd "$RL" && python3 -m pytest -q tests
}
one_sim_check() {
    if pgrep -x orca_rl_sim >/dev/null; then
        echo "WARNING: an orca_rl_sim is already running (two sims on one ROS_DOMAIN_ID mix their data)."
        echo "         run '$0 stop' first, or set a different ROS_DOMAIN_ID."
    fi
}

cmd="${1:-run}"; shift || true
case "$cmd" in
  build)     build ;;
  run)       need_build; one_sim_check; ros2 launch "$LAUNCH" world:="${1:-rl}" ;;
  watch)     need_build; one_sim_check; ros2 launch "$LAUNCH" world:="${1:-rl}" follow:=true ;;
  headless)  need_build; one_sim_check; ros2 launch "$LAUNCH" world:="${1:-rl}" headless:=true rtf:="${2:-3}" ;;
  autopilot) need_build; one_sim_check; ros2 launch "$LAUNCH" world:="${1:-rl}" mode:=autopilot ;;
  legacy)    need_build; one_sim_check; ros2 launch "$LAUNCH" world:="${1:-rl}" policy_path:="$RL/logs/trained_policy.npz" ;;
  debug)     need_build; one_sim_check; ros2 launch "$LAUNCH" world:="${1:-rl}" nav_mode:=DEBUG_GROUND_TRUTH ;;
  manual)    need_build; one_sim_check; cd "$SIM_BUILD" && ./orca_rl_sim --world "${1:-rl}" &
             sleep 5
             echo "Stonefish is up. In another terminal try:"
             echo "  python3 $RL/tools/manual_cmd.py --speed 0.6"
             echo "  python3 $RL/tools/manual_cmd.py --speed 0.5 --yaw-rate 0.03"
             echo "  ros2 service call /orca/reset std_srvs/srv/Trigger"
             wait ;;
  topics)    python3 "$RL/tools/check_stonefish_topics.py" --duration "${1:-8}" ;;
  eval)      need_build; one_sim_check
             "$HERE/scripts/closed_loop_eval.sh" "$RL/logs/trained_policy_obstacle.npz" "${1:-3}" "${2:-1800}" rl survey multisub
             for f in "$RL"/logs/stonefish_runs/*.jsonl; do python3 "$RL/tools/plot_stonefish_run.py" "$f"; done
             python3 "$RL/evaluate_obstacle.py" --seeds 7000 100 ;;
  train)     cd "$RL" && python3 -u train_obstacle.py --generations 200 ;;
  docs)      python3 "$HERE/docs/make_figures.py" && python3 "$HERE/docs/build_presentation.py" \
                 && python3 "$HERE/docs/report/build_report.py" ;;
  stop)      pkill -x orca_rl_sim && echo "stopped" || echo "nothing running" ;;
  *)         sed -n '2,17p' "$0"; exit 1 ;;
esac

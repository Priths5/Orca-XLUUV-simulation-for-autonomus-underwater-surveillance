"""Save the client area of the Stonefish window (no title bar) - used by capture_screenshots.sh."""
import re
import subprocess
import sys

from PIL import ImageGrab

title = sys.argv[2] if len(sys.argv) > 2 else "Orca XLUUV - RL over ROS 2"
info = subprocess.run(["xwininfo", "-name", title], capture_output=True, text=True).stdout
g = {k: int(v) for k, v in re.findall(r"(Absolute upper-left [XY]|Width|Height):\s+(-?\d+)", info)}
if len(g) < 4:
    sys.exit("Stonefish window not found")
x, y = g["Absolute upper-left X"], g["Absolute upper-left Y"]
ImageGrab.grab(bbox=(x + 12, y + 38, x + g["Width"] - 12, y + g["Height"] - 18)).save(sys.argv[1])
print("saved", sys.argv[1])

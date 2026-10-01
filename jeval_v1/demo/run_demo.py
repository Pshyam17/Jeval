#!/usr/bin/env python3.12
"""
Start the Jeval live demo.

    python3.12 demo/run_demo.py                  # normal speed
    python3.12 demo/run_demo.py --speed 0.5      # half speed for explanation
    python3.12 demo/run_demo.py --speed 2.0      # double speed
    python3.12 demo/run_demo.py --port 8765
    python3.12 demo/run_demo.py --no-browser
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def wait_for_ready(port: int, timeout: float = 120.0) -> bool:
    """Poll /ready until NIM warmup completes (prints dots during wait)."""
    deadline = time.time() + timeout
    print("warming up NIM", end="", flush=True)
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/ready", timeout=2)
            print("\nNIM warm — opening browser", flush=True)
            return True
        except Exception:
            print(".", end="", flush=True)
            time.sleep(1.0)
    return False


def main():
    p = argparse.ArgumentParser(description="Start the Jeval live demo.")
    p.add_argument("--speed",      type=float, default=1.0,
                   help="Replay speed multiplier (default 1.0)")
    p.add_argument("--port",       type=int,   default=8765,
                   help="HTTP port (default 8765)")
    p.add_argument("--no-browser", action="store_true",
                   help="Skip auto-opening the browser")
    args = p.parse_args()

    env = os.environ.copy()
    env["REPLAY_SPEED"] = str(args.speed)

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "demo.server:app",
         "--port", str(args.port), "--log-level", "warning"],
        env=env,
    )

    print(f"starting Jeval demo on http://localhost:{args.port} ...")
    if not wait_for_ready(args.port, timeout=120.0):
        print("NIM did not warm up in time")
        proc.terminate()
        sys.exit(1)

    if not args.no_browser:
        import webbrowser
        webbrowser.open(f"http://localhost:{args.port}")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nstopped")


if __name__ == "__main__":
    main()

"""One-click launcher: set up, train if needed, and open the dashboard.

    python3 run_project.py

1. Creates the .venv virtual environment and installs requirements (first run only).
2. Trains the models with main.py if they are missing (about 2 minutes, first run only).
3. Starts the Streamlit dashboard and prints the links to open it.
Press Ctrl+C to stop.
"""
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQUIRED_FILES = [ROOT / "models/module1_release_rate.joblib", ROOT / "models/module1_cumulative_release.joblib",
                  ROOT / "models/module2_recommender.joblib", ROOT / "outputs/results_summary.json"]


def step(msg):
    print(f"\n>>> {msg}", flush=True)


def ensure_venv():
    """Re-run this script with the project's virtual environment, creating it if needed."""
    if Path(sys.prefix).resolve() == VENV.resolve():
        return
    if not VENV_PY.exists():
        step("First run: creating the virtual environment and installing packages (a few minutes)")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
        subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "-q", "--upgrade", "pip"])
        subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])
    os.execv(str(VENV_PY), [str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])


def ensure_packages():
    try:
        import sklearn, streamlit, xgboost  # noqa: F401
    except ImportError:
        step("Installing missing packages")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])
    except Exception as e:  # xgboost raises XGBoostError when the OpenMP runtime is missing
        if sys.platform == "darwin":
            sys.exit(f"XGBoost could not load ({type(e).__name__}). Install OpenMP with:  brew install libomp")
        raise


def ensure_models():
    if all(p.exists() for p in REQUIRED_FILES):
        return
    step("Models not found: training them now with main.py (about 2 minutes)")
    subprocess.check_call([sys.executable, str(ROOT / "main.py")], cwd=ROOT)


def free_port(start=8501):
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port found between 8501 and 8550")


def lan_ip():
    """This computer's address on the local network (for other devices on the same Wi-Fi)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return None


def wait_until_up(url, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/_stcore/health", timeout=2) as r:
                if r.read().strip() == b"ok":
                    return True
        except OSError:
            time.sleep(1)
    return False


def main():
    ensure_venv()
    ensure_packages()
    ensure_models()

    port = free_port()
    local = f"http://localhost:{port}"
    ip = lan_ip()
    step("Starting the dashboard")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard.py"), "--server.port", str(port),
         "--server.address", "0.0.0.0", "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    if not wait_until_up(local):
        proc.terminate()
        sys.exit("The dashboard did not start:\n" + proc.stderr.read().decode(errors="replace")[-2000:])

    line = "=" * 64
    print(f"\n{line}\n  SMART FERTILIZER DASHBOARD IS RUNNING\n{line}")
    print(f"  On this computer:          {local}")
    if ip:
        print(f"  Other devices (same Wi-Fi): http://{ip}:{port}")
    print(f"{line}\n  Keep this window open. Press Ctrl+C to stop.\n", flush=True)
    webbrowser.open(local)

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping the dashboard...")
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()

"""agriGPT Native Desktop Application Launcher.
Starts the local AgriGPT engine and launches a dedicated standalone desktop app window
(frameless app mode without browser URL bar or tabs).
"""

import os
import sys
import time
import subprocess
import socket
import urllib.request
import webbrowser
from pathlib import Path

APP_NAME = "agriGPT"
APP_PORT = 8501
APP_URL = f"http://localhost:{APP_PORT}"
BASE_DIR = Path(__file__).resolve().parent

def is_server_running(host="localhost", port=APP_PORT) -> bool:
    """Check if the Streamlit server is already responding."""
    try:
        req = urllib.request.urlopen(f"http://{host}:{port}/_stcore/health", timeout=1)
        return req.getcode() == 200
    except Exception:
        return False

def find_browser_app_mode():
    """Locate Chrome or Edge to run in native App Mode."""
    candidate_paths = [
        # Microsoft Edge
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Microsoft/Edge/Application/msedge.exe",
        # Google Chrome
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Edge/Application/msedge.exe"
    ]
    for p in candidate_paths:
        if p.exists():
            return str(p)
    return None

def start_streamlit_server():
    """Launch Streamlit in the background."""
    app_py = BASE_DIR / "app.py"
    cmd = [
        sys.executable,
        "-m", "streamlit", "run",
        str(app_py),
        f"--server.port={APP_PORT}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    # Hide console window on Windows if possible
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW

    proc = subprocess.Popen(cmd, cwd=str(BASE_DIR), creationflags=creationflags)
    return proc

def launch_app_window():
    """Open AgriGPT in a dedicated native application window."""
    browser_exe = find_browser_app_mode()
    app_profile_dir = BASE_DIR / "assets" / "app_profile"
    os.makedirs(app_profile_dir, exist_ok=True)

    if browser_exe:
        # Launch dedicated frameless native app window
        cmd = [
            browser_exe,
            f"--app={APP_URL}",
            f"--user-data-dir={app_profile_dir}",
            "--disable-extensions",
            f"--app-id={APP_NAME}"
        ]
        subprocess.Popen(cmd)
    else:
        # Fallback to default browser
        webbrowser.open(APP_URL)

def main():
    print(f"[*] Initializing {APP_NAME}...")

    # 1. Start server if not already running
    server_proc = None
    if not is_server_running():
        print(f"[*] Starting {APP_NAME} local engine on port {APP_PORT}...")
        server_proc = start_streamlit_server()

        # Wait up to 15 seconds for server to be ready
        retries = 30
        while retries > 0:
            if is_server_running():
                print(f"[+] {APP_NAME} engine online!")
                break
            time.sleep(0.5)
            retries -= 1
    else:
        print(f"[+] {APP_NAME} engine already running.")

    # 2. Launch dedicated App Window
    print(f"[*] Opening {APP_NAME} window...")
    launch_app_window()
    print(f"[+] {APP_NAME} launched successfully.")

if __name__ == "__main__":
    main()

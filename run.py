import os
import platform
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import shutil

#checking for path and OS
BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / "venv"

IS_WINDOWS = platform.system() == "Windows"
PYTHON_BIN = VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")

STREAMLIT_APP = "app.py"
STREAMLIT_PORT = "8501"


# check for python
def find_system_python():
    """Find a working Python 3 interpreter."""
    for cmd in ("python", "python3"):
        if shutil.which(cmd):
            try:
                out = subprocess.check_output([cmd, "--version"], stderr=subprocess.STDOUT)
                if b"Python 3" in out:
                    return cmd
            except Exception:
                pass

    sys.exit("Python 3 not found. Please install Python 3.8+.")


# venv setup
def create_virtualenv():
    if not PYTHON_BIN.exists():
        print("Creating virtual environment...")
        system_python = find_system_python()
        subprocess.check_call([system_python, "-m", "venv", str(VENV_DIR)])
    else:
        print("Virtual environment already exists.")



def install_dependencies():
    if not PYTHON_BIN.exists():
        print("Virtual environment not found. Cannot install dependencies.")
        return

    print("Installing dependencies (if needed)...")
    subprocess.check_call([str(PYTHON_BIN), "-m", "pip", "install", "--upgrade", "pip"])

    req_file = BASE_DIR / "requirements.txt"
    if req_file.exists():
        subprocess.check_call(
            [str(PYTHON_BIN), "-m", "pip", "install", "-r", str(req_file)]
        )
    else:
        print("No requirements.txt found — skipping install.")



# Run Streamlit
def run_streamlit():
    print(f"\nStarting Streamlit app on http://localhost:{STREAMLIT_PORT} ...\n")

    proc = subprocess.Popen([
        str(PYTHON_BIN),
        "-m",
        "streamlit",
        "run",
        STREAMLIT_APP,
        "--server.port",
        STREAMLIT_PORT,
        "--server.headless",
        "true"
    ])

    time.sleep(3)
    webbrowser.open(f"http://localhost:{STREAMLIT_PORT}")

    #display running
    print(" Streamlit app is running!")
    print(f"URL: http://localhost:{STREAMLIT_PORT}")
    
    print("\nPress CTRL + C to stop the application.\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down Streamlit app...")
        proc.terminate()
        print("Application stopped.")


#main execution
if __name__ == "__main__":
    create_virtualenv()
    install_dependencies()
    run_streamlit()
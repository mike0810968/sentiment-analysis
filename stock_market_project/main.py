"""Entry point for launching the Streamlit dashboard."""

import os
import subprocess
import sys


def main():
    project_root = os.path.abspath(os.path.dirname(__file__))
    dashboard_path = os.path.join(project_root, "app", "dashboard.py")

    print("Starting Streamlit dashboard...")

    # Set OpenMP environment variables for XGBoost
    env = os.environ.copy()
    env['LDFLAGS'] = "-L/opt/homebrew/opt/libomp/lib"
    env['CPPFLAGS'] = "-I/opt/homebrew/opt/libomp/include"

    # Run streamlit using python -m to ensure it uses the correct Python environment
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path],
                  env=env, check=True)


if __name__ == "__main__":
    main()

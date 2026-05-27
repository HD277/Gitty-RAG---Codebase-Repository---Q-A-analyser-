"""
main.py — Run this file to start the whole application.

Usage:
  python main.py
"""

import os
import sys
import subprocess
from pathlib import Path

def check_env():
    """Make sure the .env file exists and has an API key."""
    env_file = Path(".env")
    if not env_file.exists():
        print("──────────────────────────────────────────────────")
        print(" .env file not found!")
        print(" Copy .env.example to .env and add your API key.")
        print("──────────────────────────────────────────────────")
        sys.exit(1)

    # Load the .env file manually
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

    if os.environ.get("GEMINI_API_KEY", "").startswith("your_"):
        print("──────────────────────────────────────────────────")
        print(" Please set your real GEMINI_API_KEY in .env")
        print("──────────────────────────────────────────────────")
        sys.exit(1)

    print("[✓] Environment loaded")

if __name__ == "__main__":
    check_env()
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   🐱 Gitty — GitHub Repo Reader                  ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print("[→] Starting API server on http://localhost:8000")
    print("[→] Open frontend/index.html in your browser to use Gitty")
    print("[→] Press Ctrl+C to stop")
    print()

    # Start FastAPI server
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

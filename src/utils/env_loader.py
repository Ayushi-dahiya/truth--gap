"""
TruthGap Environment Loader
Loads environment variables from .env file into os.environ.
"""

import os
from pathlib import Path


def load_env(env_path: Path = None):
    """
    Search for and load .env file into os.environ.
    Looks in current directory, project root, and parent directories.
    """
    if env_path is None:
        candidates = [
            Path(".env"),
            Path(__file__).resolve().parent.parent.parent / ".env",
            Path(__file__).resolve().parent.parent / ".env",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                env_path = candidate
                break

    if env_path and env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("\"'").strip()
                        if key and key not in os.environ:
                            os.environ[key] = val
        except Exception as e:
            print(f"[Warning] Failed to load .env file: {e}")

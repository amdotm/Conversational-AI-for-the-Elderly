from __future__ import annotations
import subprocess

def play_wav(path: str) -> None:
    subprocess.run(["paplay", path], check=False)

"""Development validation entrypoint, intentionally independent of hardware/ML checkpoints."""
import subprocess
import sys

if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q"]))

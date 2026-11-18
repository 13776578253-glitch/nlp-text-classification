"""Stream training events to the terminal and a Markdown progress snapshot."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


def write_atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")
    temp.replace(path)


class Progress:
    def __init__(self, path: Path, command: str, title: str = "Batch A training progress") -> None:
        self.path = path
        self.command = command
        self.title = title
        self.context = {}
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, phase: str, message: str, **details) -> None:
        with self._lock:
            self._emit(phase, message, **details)

    def _emit(self, phase: str, message: str, **details) -> None:
        details = {**self.context, **details}
        stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        payload = {"time": stamp, "phase": phase, "message": message, **details}
        run = f" [{details['run_id']}]" if "run_id" in details else ""
        position = f" [{details['matrix_position']}]" if "matrix_position" in details else ""
        print(f"[{stamp}]{position}{run} {phase} {message}", flush=True)
        with self.path.with_suffix(".jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        lines = [f"# {self.title}", "", f"Updated {stamp}", "",
                 f"**{phase}** {message}", "", f"Command: `{self.command}`"]
        lines.extend(f"- {name}: `{value}`" for name, value in details.items())
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for _ in range(4):
            try:
                tmp.replace(self.path)
                break
            except PermissionError:
                time.sleep(0.15)
        else:
            print(f"Progress snapshot currently in use: {self.path}", flush=True)

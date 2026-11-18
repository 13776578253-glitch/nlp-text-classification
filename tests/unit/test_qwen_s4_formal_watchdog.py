import json
from types import SimpleNamespace

import pytest
import torch

from nlp_text_classification.experiments import batch_b_s4_formal as formal
from nlp_text_classification.experiments import batch_b_s4_formal_watchdog as watchdog


def test_formal_vram_warning_does_not_abort_on_transient_dip(monkeypatch):
    values = iter([2400, 1700, 3000, 2500, 1700])
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (next(values) * formal.MIB, 1))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    events = []
    progress = SimpleNamespace(emit=lambda phase, _message, **data: events.append((phase, data)))
    monitor = SimpleNamespace(samples=[])
    gate = formal.GpuHeadroom(monitor, progress, strict=False)
    for i in range(4):
        gate.check(f"update {i}")
    assert gate.min_free_mib == 1700
    assert [data["recent_free_mib"] for phase, data in events if phase == "VRAM_WARNING"] == [
        2400, 2500,
    ]
    with pytest.raises(RuntimeError, match="1700 MiB is below 2048 MiB"):
        formal.GpuHeadroom(monitor, progress, strict=True).check("bounded preflight")


def test_watchdog_retries_once_from_last_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoints" / "step-00000500" / "checkpoint.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}", encoding="utf-8")
    invocations = []

    def run(command, _root, _limits, *, protect_vram):
        invocations.append((command, protect_vram))
        return watchdog.CUDA_OOM_EXIT, True

    monkeypatch.setattr(watchdog, "_run_worker", run)
    monkeypatch.setattr(watchdog, "_wait_for_release", lambda *_: 12000)
    entry = ["train", "--seed", "42", "--preflight", "prelaunch-v5"]
    assert watchdog.supervise(entry, tmp_path) == watchdog.CUDA_OOM_EXIT
    assert [cmd[-len(entry)] for cmd, _ in invocations] == ["train", "resume"]
    assert all(protect for _, protect in invocations)
    event = json.loads((tmp_path / "watchdog.jsonl").read_text(encoding="utf-8"))
    assert event["event"] == "RETRY" and event["free_mib"] == 12000
    assert checkpoint.read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize("mode", ["train", "preflight"])
def test_watchdog_requires_gpu_release_and_keeps_preflight_one_shot(monkeypatch, tmp_path, mode):
    invocations = []
    release_checks = []

    def run(command, _root, _limits, *, protect_vram):
        invocations.append((command, protect_vram))
        return watchdog.WATCHDOG_EXIT, True

    monkeypatch.setattr(watchdog, "_run_worker", run)
    monkeypatch.setattr(watchdog, "_wait_for_release", lambda *_: release_checks.append(True))
    assert watchdog.supervise([mode], tmp_path) == watchdog.WATCHDOG_EXIT
    assert len(invocations) == 1 and invocations[0][1] == (mode == "train")
    assert len(release_checks) == (1 if mode == "train" else 0)


def test_watchdog_ignores_old_or_partial_progress(tmp_path):
    journal = tmp_path / "progress.jsonl"
    journal.write_text('{"pid":12,"phase":"LOAD_MODEL"}\n{"pid":38,"phase":"TRAIN"}\n'
                       '{"pid":12', encoding="utf-8")
    assert watchdog._last_progress(journal, 12)["phase"] == "LOAD_MODEL"
    assert watchdog._last_progress(journal, 38)["phase"] == "TRAIN"
    assert watchdog._last_progress(journal, 17) is None

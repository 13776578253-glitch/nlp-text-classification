from threading import Event

import pytest

from nlp_text_classification.cli import main
from nlp_text_classification.training import resources


def test_batch_needs_physical_and_commit_headroom(monkeypatch):
    for values in ((7, 12), (12, 7)):
        memory = resources.AvailableMemory(*(value * resources.GIB for value in values))
        monkeypatch.setattr(resources, "memory_available", lambda info=memory: info)
        with pytest.raises(RuntimeError, match="startup needs 8 GiB"):
            resources.require_headroom()


def test_online_memory_pressure_requests_interrupt(monkeypatch):
    states = iter([resources.AvailableMemory(10 * resources.GIB, 12 * resources.GIB),
                   resources.AvailableMemory(5 * resources.GIB, 12 * resources.GIB)])
    interrupted = Event()
    messages = []

    class Log:
        def emit(self, phase, message, **details):
            messages.append(phase)

    monkeypatch.setattr(resources, "memory_available", lambda: next(states))
    monkeypatch.setattr(resources, "set_windows_process_limit", lambda limit: None)
    monkeypatch.setattr(resources._thread, "interrupt_main", interrupted.set)
    with resources.protect(Log()):
        assert interrupted.wait(4)
    assert messages == ["RESOURCES", "PAUSED_MEMORY"]


def test_training_cli_exits_before_dispatch_when_memory_is_low(monkeypatch):
    monkeypatch.setattr("sys.argv", ["nlp-text-classification", "batch-a", "run"])
    memory = resources.AvailableMemory(resources.GIB, 3 * resources.GIB)
    monkeypatch.setattr(resources, "memory_available", lambda: memory)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 75

"""Isolated HTTP inference worker for Batch B S4 Qwen3.5-4B QLoRA checkpoints (WSL2)."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

QWEN_FREEZE_REL = Path(
    "reports/batch-b-s4/formal/frozen/"
    "97ce29261f3844e715731f62f7859df2397db2e7381744bc8a0db86cff4e5d15.json"
)
QWEN_CANDIDATE_REL = Path("reports/batch-b-s4/performance/recommended-training.json")


def tokenize_qwen_single(tokenizer: Any, text: str, max_length: int, pad_token_id: int) -> dict[str, Any]:
    """Tokenize a single text using the frozen Qwen S3/S4 right-padded multiple-of-8 contract."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input text must not be empty")
    raw_ids = tokenizer(
        cleaned,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )["input_ids"]
    if not raw_ids:
        raise ValueError("Tokenizer produced zero tokens for input text")
    if pad_token_id in raw_ids:
        raw_ids = [tok for tok in raw_ids if tok != pad_token_id]
        if not raw_ids:
            raise ValueError("Input contains only PAD/EOS tokens")

    truncated = len(raw_ids) > max_length
    kept = raw_ids[:max_length]
    padded_len = min(max_length, max(8, int(math.ceil(len(kept) / 8) * 8)))
    pad_len = padded_len - len(kept)
    input_ids = kept + [pad_token_id] * pad_len
    attention_mask = [1] * len(kept) + [0] * pad_len
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_count": len(raw_ids),
        "truncated": truncated,
        "preview_tokens": tokenizer.convert_ids_to_tokens(kept[:24]),
    }


class QwenService:
    """Lazy-loaded NF4 Qwen3.5-4B backbone with multi-seed LoRA adapter switching."""

    def __init__(self, root: Path | str = ".") -> None:
        self.root = Path(root).resolve()
        self.freeze = json.loads((self.root / QWEN_FREEZE_REL).read_text(encoding="utf-8"))
        candidate = json.loads((self.root / QWEN_CANDIDATE_REL).read_text(encoding="utf-8"))
        self.recipe = self.freeze["recipe"]
        self.protocol = candidate["protocol"]
        self.labels: list[str] = list(self.freeze["label_order"])
        self.max_length = int(self.recipe["max_length"])
        self.pad_token_id = int(self.protocol["preprocessing"]["pad_token_id"])
        self.snapshot = (
            self.root / "artifacts/batch-b-s3/official" / self.recipe["model_revision"]
        )
        self.seed_entries = {int(r["seed"]): r for r in self.freeze["runs"]}
        self._lock = threading.RLock()
        self._tokenizer: Any = None
        self._model: Any = None
        self._loaded_adapters: set[int] = set()
        self._active_seed: int | None = None

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from nlp_text_classification.experiments.batch_b_resources import load_tokenizer

            self._tokenizer = load_tokenizer(self.snapshot)
        return self._tokenizer

    def _ensure_model(self, seed: int) -> Any:
        if seed not in self.seed_entries:
            raise KeyError(f"Unsupported Qwen S4 seed: {seed}")
        entry = self.seed_entries[seed]
        adapter_dir = self.root / entry["run"] / entry["checkpoint"] / "adapter"
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"Adapter directory missing: {adapter_dir}")

        if self._model is None:
            from peft import PeftModel

            from nlp_text_classification.training.qwen_s3 import load_base

            base_model, _ = load_base(self.snapshot, self.protocol, self.labels, audit=False)
            for param in base_model.parameters():
                param.requires_grad = False
            self._model = PeftModel.from_pretrained(
                base_model,
                adapter_dir,
                adapter_name=f"seed-{seed}",
                is_trainable=False,
            )
            self._model.eval()
            self._loaded_adapters = {seed}
            self._active_seed = seed
            return self._model

        if seed not in self._loaded_adapters:
            try:
                self._model.load_adapter(
                    str(adapter_dir),
                    adapter_name=f"seed-{seed}",
                    is_trainable=False,
                )
                self._loaded_adapters.add(seed)
            except (RuntimeError, ValueError, KeyError, TypeError):
                # Fallback: clean reload if score head modules_to_save collides across adapters
                import torch

                del self._model
                self._model = None
                self._loaded_adapters.clear()
                self._active_seed = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._ensure_model(seed)

        if self._active_seed != seed:
            self._model.set_adapter(f"seed-{seed}")
            self._active_seed = seed
        return self._model

    def predict(self, text: str, seed: int = 42) -> dict[str, Any]:
        import torch

        with self._lock:
            started = time.perf_counter()
            tokenizer = self._ensure_tokenizer()
            tok = tokenize_qwen_single(tokenizer, text, self.max_length, self.pad_token_id)
            model = self._ensure_model(seed)

            input_ids = torch.tensor([tok["input_ids"]], dtype=torch.long, device="cuda")
            attention_mask = torch.tensor([tok["attention_mask"]], dtype=torch.long, device="cuda")
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    ).logits
                scores = logits.float().sigmoid()[0].detach().cpu().numpy().tolist()

            latency_ms = (time.perf_counter() - started) * 1000.0
            return {
                "seed": seed,
                "scores": [float(s) for s in scores],
                "threshold": float(self.seed_entries[seed]["threshold"]),
                "latency_ms": round(latency_ms, 2),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cuda",
                "preprocessing": {
                    "segmentation_applied": False,
                    "token_count": tok["token_count"],
                    "truncated": tok["truncated"],
                    "preview_tokens": tok["preview_tokens"],
                },
            }

    def health(self) -> dict[str, Any]:
        import torch

        vram_free = None
        device = None
        if torch.cuda.is_available():
            free_b, _ = torch.cuda.mem_get_info()
            vram_free = round(free_b / (1024**2), 1)
            device = torch.cuda.get_device_name(0)
        return {
            "status": "ok",
            "model_id": "goemotions/Qwen3.5-4B",
            "loaded_seed": self._active_seed,
            "loaded_seeds": sorted(self._loaded_adapters),
            "available_seeds": sorted(self.seed_entries),
            "device": device,
            "vram_free_mib": vram_free,
        }


def make_handler(service: QwenService):
    class QwenRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(HTTPStatus.OK, service.health())
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})

        def do_POST(self) -> None:
            if self.path != "/predict":
                self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                text = str(body.get("text", ""))
                seed = int(body.get("seed", 42))
                result = service.predict(text, seed=seed)
                self._send_json(HTTPStatus.OK, result)
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"detail": str(exc)})
            except ValueError as exc:
                self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": str(exc)})
            except (RuntimeError, OSError, TypeError, json.JSONDecodeError) as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return QwenRequestHandler


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    parser = argparse.ArgumentParser(description="EmotionLab isolated WSL2 Qwen S4 worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--preload-seed", type=int, default=None)
    args = parser.parse_args()

    service = QwenService(root=args.root)
    if args.preload_seed is not None:
        print(f"Preloading Qwen3.5-4B QLoRA adapter seed-{args.preload_seed}...", flush=True)
        with service._lock:
            service._ensure_tokenizer()
            service._ensure_model(args.preload_seed)
        print("Qwen3.5-4B ready.", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"Qwen worker listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

"""Fetch and verify only the pinned official S3 model files."""

from pathlib import Path

from nlp_text_classification.experiments.batch_b_resources import _verify_download
from nlp_text_classification.experiments.suite import checked_file, load_yaml, read_json
from nlp_text_classification.training.progress import write_atomic_json

SUITE = Path("configs/experiment_suites/batch_b_s3.yaml")
ROOT = Path("artifacts/batch-b-s3")
MANIFEST = ROOT / "model-resources.json"
SHARDS = {f"model.safetensors-{part:05d}-of-00002.safetensors" for part in (1, 2)}
FILES = SHARDS | {"config.json", "model.safetensors.index.json", "tokenizer.json",
                  "tokenizer_config.json", "vocab.json", "merges.txt", "README.md", "LICENSE"}


def download(config):
    from huggingface_hub import HfApi, snapshot_download

    repo, revision = config["model_id"], config["model_revision"]
    root = ROOT / "official" / revision
    hub = HfApi().model_info(repo, revision=revision, files_metadata=True, token=False, timeout=30)
    if (hub.id, hub.sha, hub.card_data.license) != (repo, revision, config["license"]):
        raise ValueError("Official Hub identity or license changed")
    files = {item.rfilename: {
        "bytes": item.size,
        **({"lfs_sha256": item.lfs.sha256} if item.lfs else {"git_blob": item.blob_id}),
    } for item in hub.siblings if item.rfilename in FILES}
    if files.keys() != FILES or any(not files[path].get("lfs_sha256") for path in SHARDS):
        raise ValueError("Official snapshot lacks required files or LFS identities")
    ROOT.mkdir(parents=True, exist_ok=True)
    metadata = {"model_id": repo, "revision": revision, "license": config["license"],
                "snapshot": root.as_posix(), "files": files}
    write_atomic_json(ROOT / "hub-metadata.json", metadata)
    print(f"Downloading {repo}@{revision} to {root}", flush=True)
    snapshot_download(repo, revision=revision, local_dir=root, cache_dir=ROOT / "hub-cache",
                      allow_patterns=sorted(FILES), token=False, max_workers=2)
    receipt = metadata | {"files": _verify_download(root, metadata)}
    write_atomic_json(MANIFEST, receipt)
    print(f"Verified {len(FILES)} files against official identities", flush=True)
    return receipt


def verify_resources(config):
    audited = read_json(MANIFEST)
    expected = ROOT / "official" / config["model_revision"]
    if (audited["model_id"], audited["revision"], audited["license"], Path(audited["snapshot"])) != (
        config["model_id"], config["model_revision"], config["license"], expected
    ) or audited["files"].keys() != FILES:
        raise ValueError("Snapshot manifest differs from the S3 contract")
    for name, item in audited["files"].items():
        checked_file(expected / name, item["sha256"])
    return expected, audited


if __name__ == "__main__":
    download(load_yaml(SUITE))

"""Download Qwen3-4B-Instruct-2507 onto the AutoDL data disk. No GPU required."""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LOCAL_DIR = Path(os.environ.get("FINTOOL_MODEL_DIR", "/root/autodl-tmp/models/Qwen3-4B-Instruct-2507"))


def _assert_data_disk(path: Path) -> None:
    resolved = path.resolve()
    if not str(resolved).startswith("/root/autodl-tmp/"):
        raise SystemExit(f"refusing to download outside data disk: {resolved}")


def _has_weights(path: Path) -> bool:
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def _try_modelscope(dest: Path) -> None:
    from modelscope.hub.snapshot_download import snapshot_download

    snapshot_download(MODEL_ID, local_dir=str(dest))


def _try_hf_mirror(dest: Path) -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, local_dir=str(dest), resume_download=True)


def main() -> None:
    _assert_data_disk(LOCAL_DIR)
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    if _has_weights(LOCAL_DIR):
        print(f"weights_already_present {LOCAL_DIR}")
        return
    errors: list[str] = []
    for name, fn in (("modelscope", _try_modelscope), ("hf-mirror", _try_hf_mirror)):
        try:
            print(f"trying_{name}", flush=True)
            fn(LOCAL_DIR)
            if _has_weights(LOCAL_DIR):
                print(f"downloaded_via_{name} {LOCAL_DIR}")
                return
            errors.append(f"{name}: finished without weight files")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"failed_{name} {exc}", file=sys.stderr, flush=True)
    raise SystemExit("model download failed:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()

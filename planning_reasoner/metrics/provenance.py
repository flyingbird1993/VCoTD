"""Reproducibility manifest helpers for every experiment run."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(cwd: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(
    config: Mapping[str, Any],
    seed: int,
    project_root: str | Path,
    files: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    hashes = {}
    for name, path in (files or {}).items():
        file_path = Path(path)
        hashes[name] = {"path": str(file_path.resolve()), "sha256": sha256_file(file_path)}
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config": dict(config),
        "files": hashes,
        "source_revision": _git_revision(project_root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "hostname": platform.node(),
            "pid": os.getpid(),
        },
    }


def write_manifest(output_dir: str | Path, manifest: Mapping[str, Any]) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True, sort_keys=True)
    return manifest_path


def record_artifacts(
    output_dir: str | Path,
    artifacts: Mapping[str, str | Path],
) -> Path:
    """Add checkpoint/prediction hashes after artifacts have been written."""
    output_path = Path(output_dir)
    manifest_path = output_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = {
        name: {
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(path),
        }
        for name, path in artifacts.items()
    }
    return write_manifest(output_path, manifest)

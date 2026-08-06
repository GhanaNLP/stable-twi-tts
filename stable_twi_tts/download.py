"""Fetch the published voice using nothing but the standard library.

`huggingface_hub` would do this in three lines, but it brings twelve transitive packages —
httpx, anyio, fsspec, hf-xet, PyYAML, tqdm and friends — to download four files totalling
77 MB. For a package whose whole premise is being lighter than a PyTorch install, that is the
wrong trade, so the default path is `urllib` against GitHub release assets and needs no
dependency at all. The Hub remains available to anyone who has `huggingface_hub` installed and
wants a specific revision.

Rolling your own downloader means owning the parts a library would have handled, so:

  **Checksums are verified, not assumed.** A truncated `model.onnx` is the failure that
  matters: onnxruntime reports it as a protobuf parse error deep in its own C++, which reads
  like a corrupt model rather than a bad download. Each file's SHA-256 is pinned below and
  checked after transfer.

  **Writes are atomic.** Download to a temporary name in the destination directory, then
  `os.replace`. An interrupted run leaves no half-file that a later run would trust — which
  matters more here than usual, since the cache is keyed by filename and has no manifest.

  **Nothing is re-downloaded.** A file whose checksum already matches is left alone, so
  `from_pretrained()` is cheap on every call after the first.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

# Pinned to a tag rather than to `latest`, so a new release cannot silently change which
# weights an installed version loads. The model was measured at this revision; the front-end
# conventions and voices.json schema are tied to it.
RELEASE_TAG = "model-v0.1.0"
BASE_URL = f"https://github.com/GhanaNLP/stable-twi-tts/releases/download/{RELEASE_TAG}"

# name -> (sha256, bytes). Regenerate with tools/publish_model_release.py, which writes this
# table from the files it uploads so the two cannot drift.
FILES: dict[str, tuple[str, int]] = {
    "model.onnx": ("4cfe48851a0dd7cb228948294aac9b7278b8f43185ec2a23ef36c68f59b45182", 80236649),
    "config.json": ("b22fbb084f656d83719061179f8f6e954f1c5a6169f84c83373e2d6f4d37c96d", 34718),
    "voices.json": ("8b11e643351a1c1315a963337c9daab955accffa101342b1242531828779de16", 4949),
    "tokens.txt": ("d6752d8b93071feb413c0538654ea8f51d088ebf2a0431a24adb4201e4bf3d79", 1412),
}


class DownloadError(RuntimeError):
    pass


def default_cache_dir() -> Path:
    """Respect XDG_CACHE_HOME, and Windows' LOCALAPPDATA, before falling back to ~/.cache."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "stable-twi-tts" / RELEASE_TAG


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _ok(path: Path, digest: str, size: int) -> bool:
    """Cheap size check first — it rejects a truncated file without hashing 80 MB."""
    return path.is_file() and path.stat().st_size == size and _sha256(path) == digest


def _fetch(url: str, dest: Path, digest: str, size: int, quiet: bool) -> None:
    tmp = None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "stable-twi-tts"})
        with urllib.request.urlopen(req, timeout=60) as r:
            # Written into the destination directory, not /tmp: os.replace is only atomic
            # within one filesystem, and /tmp is frequently a separate mount.
            fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
            tmp = Path(tmp_name)
            done = 0
            with os.fdopen(fd, "wb") as out:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if not quiet and size > 4 << 20:
                        pct = 100 * done / size
                        print(f"\r  {dest.name}  {done/1e6:6.1f} / {size/1e6:.1f} MB"
                              f"  {pct:5.1f}%", end="", file=sys.stderr, flush=True)
        if not quiet and size > 4 << 20:
            print(file=sys.stderr)
    except urllib.error.HTTPError as e:
        if tmp and tmp.exists():
            tmp.unlink()
        raise DownloadError(f"{url} returned HTTP {e.code}") from e
    except OSError as e:                      # URLError, timeouts, disk full
        if tmp and tmp.exists():
            tmp.unlink()
        raise DownloadError(f"could not download {url}: {e}") from e

    got = _sha256(tmp)
    if got != digest:
        actual = tmp.stat().st_size
        tmp.unlink()
        raise DownloadError(
            f"{dest.name} failed its checksum — expected {digest[:12]}…, got {got[:12]}… "
            f"({actual} bytes, expected {size}). A proxy or captive portal may have served "
            f"something else. Delete {dest.parent} and retry."
        )
    os.replace(tmp, dest)


def ensure_model(cache_dir: str | Path | None = None, quiet: bool = False) -> Path:
    """Return a directory holding the voice, downloading whatever is missing.

    Safe to call repeatedly: files already present and matching their checksum are untouched.
    """
    d = Path(cache_dir) if cache_dir else default_cache_dir()
    d.mkdir(parents=True, exist_ok=True)

    missing = [(n, h, s) for n, (h, s) in FILES.items() if not _ok(d / n, h, s)]
    if not missing:
        return d

    total = sum(s for _, _, s in missing) / 1e6
    if not quiet:
        print(f"fetching the voice ({total:.0f} MB) -> {d}", file=sys.stderr)

    free = shutil.disk_usage(d).free
    need = sum(s for _, _, s in missing) * 2      # the .part file coexists with the final one
    if free < need:
        raise DownloadError(f"not enough free space in {d}: need ~{need/1e6:.0f} MB, "
                            f"have {free/1e6:.0f} MB")

    for name, digest, size in missing:
        _fetch(f"{BASE_URL}/{name}", d / name, digest, size, quiet)
    return d

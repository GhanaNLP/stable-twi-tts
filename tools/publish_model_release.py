"""Upload a voice as GitHub release assets, and rewrite download.py's checksum table from it.

The table in download.py must describe exactly the files that were uploaded. Computing it by
hand invites the one bug that is genuinely nasty here: a checksum that matches the *previous*
release, so every download fails verification and the error blames the user's network. So this
script does both halves — upload, then rewrite the table from the same bytes it sent.

Usage:
    python tools/publish_model_release.py voices/stable-twi-tts --tag model-v0.1.0

Needs the `gh` CLI, authenticated. Re-running against an existing tag replaces the assets, which
is fine before anyone depends on the tag and a bad idea afterwards — cut a new tag instead, since
an installed version pins the tag it was released with.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

FILES = ["model.onnx", "config.json", "voices.json", "tokens.txt"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("voice_dir", help="directory holding model.onnx, config.json, voices.json")
    ap.add_argument("--tag", required=True, help="release tag, e.g. model-v0.1.0")
    ap.add_argument("--repo", default="GhanaNLP/stable-twi-tts")
    ap.add_argument("--notes", default=None, help="release notes; a default is generated")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = Path(args.voice_dir)
    paths = [d / f for f in FILES]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        print(f"missing from {d}: {missing}", file=sys.stderr)
        return 1

    table = {p.name: (sha256(p), p.stat().st_size) for p in paths}
    total = sum(s for _, s in table.values()) / 1e6
    print(f"{len(paths)} files, {total:.1f} MB")
    for n, (h, s) in table.items():
        print(f"  {n:14} {s:>10}  {h[:16]}…")

    # Rewrite download.py's table first: if the upload fails the repo is unchanged in any way
    # that matters, whereas an upload followed by a failed rewrite leaves them inconsistent.
    dl = Path(__file__).resolve().parent.parent / "stable_twi_tts" / "download.py"
    src = dl.read_text(encoding="utf-8")
    body = "".join(f'    "{n}": ("{h}", {s}),\n' for n, (h, s) in table.items())
    new, count = re.subn(r"FILES: dict\[str, tuple\[str, int\]\] = \{.*?\n\}",
                         "FILES: dict[str, tuple[str, int]] = {\n" + body + "}",
                         src, count=1, flags=re.S)
    if count != 1:
        print("could not find the FILES table in download.py", file=sys.stderr)
        return 1
    new, count = re.subn(r'RELEASE_TAG = "[^"]*"', f'RELEASE_TAG = "{args.tag}"', new, count=1)
    if count != 1:
        print("could not find RELEASE_TAG in download.py", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n--dry-run: nothing uploaded, download.py unchanged")
        return 0

    dl.write_text(new, encoding="utf-8")
    print(f"\nrewrote {dl.name}: tag {args.tag}, {len(table)} checksums")

    notes = args.notes or (
        f"Inference files for stable-twi-tts.\n\n"
        f"Downloaded automatically by `from_pretrained()` over urllib, with checksums verified — "
        f"no `huggingface_hub` needed. The same weights are on the Hub at "
        f"`ghanaopendata/stable-twi-tts`, which additionally carries the training checkpoint and "
        f"the model card.\n\n"
        + "\n".join(f"- `{n}` — {s:,} bytes, `sha256:{h}`" for n, (h, s) in table.items()))

    existing = subprocess.run(["gh", "release", "view", args.tag, "--repo", args.repo],
                              capture_output=True, text=True)
    if existing.returncode == 0:
        print(f"release {args.tag} exists; replacing assets")
        cmd = ["gh", "release", "upload", args.tag, *map(str, paths),
               "--repo", args.repo, "--clobber"]
    else:
        cmd = ["gh", "release", "create", args.tag, *map(str, paths), "--repo", args.repo,
               "--title", f"Model {args.tag}", "--notes", notes]

    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("upload failed; download.py was already rewritten — revert it or retry",
              file=sys.stderr)
        return r.returncode

    print(f"\nhttps://github.com/{args.repo}/releases/tag/{args.tag}")
    print("now: bump the version in pyproject.toml, commit, and publish to PyPI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

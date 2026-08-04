"""Command line: one utterance, or a whole corpus.

Batch generation is a first-class mode, not an afterthought. Anyone building a dataset, dubbing
a corpus or evaluating coverage needs to run thousands of lines, resume after an interruption,
and get a machine-readable record of what was produced — including what failed and why.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


def _read_items(path: Path, default_voice: str | None, default_lang: str) -> list[dict]:
    """Accepts .txt (one utterance per line), .csv/.tsv (with a text column), or .jsonl.

    Per-row voice and language override the defaults, so one file can mix voices and languages.
    """
    suffix = path.suffix.lower()
    items: list[dict] = []

    if suffix == ".jsonl":
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                r = json.loads(line)
                r.setdefault("id", f"{i:06d}")
                items.append(r)
    elif suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        with open(path, encoding="utf-8", newline="") as fh:
            rd = csv.DictReader(fh, delimiter=delim)
            if not rd.fieldnames or "text" not in rd.fieldnames:
                raise SystemExit(f"{path} needs a 'text' column; found {rd.fieldnames}")
            for i, r in enumerate(rd):
                r = {k: v for k, v in r.items() if v not in (None, "")}
                r.setdefault("id", f"{i:06d}")
                items.append(r)
    else:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                items.append({"id": f"{i:06d}", "text": line.strip()})

    for r in items:
        r.setdefault("language", default_lang)
        if default_voice and "voice" not in r:
            r["voice"] = default_voice
        if "text" not in r:
            raise SystemExit(f"row {r.get('id')} has no text")
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ghana-tts",
                                description="Twi / Ghanaian English speech synthesis")
    ap.add_argument("--model", required=True, help="voice directory (model.onnx, config.json)")
    ap.add_argument("--voice", default=None, help="voice name; default is the best per language")
    ap.add_argument("--language", default="twi", choices=["twi", "eng", "mixed"],
                    help="'mixed' treats [bracketed] spans as English inside a Twi frame")
    ap.add_argument("--length-scale", type=float, default=1.0, help=">1 speaks more slowly")
    ap.add_argument("--noise-scale", type=float, default=0.667)
    ap.add_argument("--noise-w", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=None, help="onnxruntime intra-op threads")
    ap.add_argument("--english-mode", default="native", choices=["native", "adapt"],
                    help="native: pronounce English words as English (needs a model trained on "
                         "English audio). adapt: respell them in Twi orthography and pronounce "
                         "them as Twi (university -> yunibesiti), which needs no English phonemes")
    ap.add_argument("--lexicon", default=None,
                    help="adaptation lexicon; defaults to the one shipped with en-twi-pronouncer")
    ap.add_argument("--strict-adapt", action="store_true",
                    help="adapt only words in the lexicon; never fall back to rules")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--text", help="one utterance")
    g.add_argument("--input", help="batch: .txt, .csv, .tsv or .jsonl")
    g.add_argument("--list-voices", action="store_true")

    ap.add_argument("--out", default=None, help="output wav, or directory in batch mode")
    ap.add_argument("--workers", type=int, default=4, help="parallel batch workers")
    ap.add_argument("--overwrite", action="store_true", help="redo already-written wavs")
    ap.add_argument("--manifest", default=None,
                    help="batch: write a JSONL record per utterance (default: <out>/manifest.jsonl)")
    args = ap.parse_args(argv)

    from .tts import GhanaTTS
    tts = GhanaTTS(args.model, num_threads=args.threads)

    if args.list_voices:
        print(tts.voices.describe())
        print(f"\n{tts.num_speakers} speakers exist in the checkpoint; the "
              f"{len(tts.voices.voices)} above are the ones with enough training audio to "
              f"sound stable.\nAny other is reachable by raw index, e.g. --voice 42.")
        return 0

    common = dict(language=args.language, length_scale=args.length_scale,
                  noise_scale=args.noise_scale, noise_w=args.noise_w,
                  english_mode=args.english_mode, lexicon_path=args.lexicon,
                  strict_adapt=args.strict_adapt)

    if args.text:
        s = tts.synthesize(args.text, voice=args.voice, **common)
        dest = Path(args.out or "out.wav")
        s.save(dest)
        print(f"{dest}  {s.duration:.2f}s  {s.n_phonemes} phonemes  voice={s.voice}")
        return 0

    if args.input:
        src = Path(args.input)
        if not src.exists():
            raise SystemExit(f"no such file: {src}")
        items = _read_items(src, args.voice, args.language)
        outdir = Path(args.out or "synth")
        print(f"{len(items)} utterances -> {outdir}  ({args.workers} workers)")

        t0 = time.time()

        def progress(r, done, total):
            if r["status"] == "error":
                print(f"  [{done}/{total}] {r['id']}: {r['error']}", file=sys.stderr)
            elif done % 25 == 0 or done == total:
                el = time.time() - t0
                rate = done / max(el, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                print(f"  [{done}/{total}] {rate:.1f}/s  eta {eta/60:.1f} min", flush=True)

        results = tts.synthesize_batch(items, outdir, workers=args.workers,
                                       skip_existing=not args.overwrite,
                                       on_done=progress, **common)

        mpath = Path(args.manifest) if args.manifest else outdir / "manifest.jsonl"
        with open(mpath, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        ok = sum(r["status"] == "ok" for r in results)
        sk = sum(r["status"] == "skipped" for r in results)
        er = [r for r in results if r["status"] == "error"]
        secs = sum(r.get("duration", 0.0) for r in results)
        el = time.time() - t0
        print(f"\n{ok} written, {sk} skipped, {len(er)} failed in {el/60:.1f} min")
        print(f"{secs/3600:.2f} h of audio  ({secs/max(el,1e-6):.0f}x realtime)")
        print(f"manifest: {mpath}")
        if er:
            print(f"\nfirst failures:")
            for r in er[:5]:
                print(f"  {r['id']}: {r['error']}")
            return 1
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Package a trained Piper checkpoint into a distributable voice directory.

Produces everything a consumer needs and nothing they don't:

    model.onnx     the exported generator
    config.json    sample rate, phoneme id map, speaker id map
    voices.json    the curated shortlist, ranked by training hours
    tokens.txt     symbol/id table, for sherpa-onnx and other native runtimes
    lexicon.txt    optional word -> phonemes, so sherpa-onnx can run without Python

The lexicon exists because of a real constraint. sherpa-onnx does its own text-to-phoneme step,
either through espeak-ng data or a lexicon file. espeak has no Asante Twi, so the only way to
get correct Twi phonemes into a native runtime is to ship the pronunciations. A lexicon covers
whatever vocabulary you generate it from and no more, which is why the Python front-end — which
calls ghana-g2p directly and handles any word, including unseen ones — stays the recommended
path wherever Python is available.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ghana_tts.voices import build_registry  # noqa: E402


def export_onnx(checkpoint: Path, out: Path, piper_python: str) -> None:
    print(f"exporting {checkpoint.name} -> {out.name}", flush=True)
    subprocess.run([piper_python, "-m", "piper.train.export_onnx",
                    "--checkpoint", str(checkpoint), "--output-file", str(out)],
                   check=True)


def write_tokens(id_map: dict[str, list[int]], path: Path) -> None:
    """tokens.txt as '<symbol> <id>' per line, ordered by id.

    Native runtimes read this instead of config.json. The space-separated format cannot express
    a symbol that *is* a space, so that entry is written as the literal word for it.
    """
    rows = sorted(((v[0], k) for k, v in id_map.items()), key=lambda r: r[0])
    with open(path, "w", encoding="utf-8") as fh:
        for idx, sym in rows:
            fh.write(f"{'<space>' if sym == ' ' else sym} {idx}\n")
    print(f"tokens.txt: {len(rows)} symbols")


def build_lexicon(manifest: Path, symbols: set[str], out: Path, dialect: str,
                  threads: int, max_words: int) -> None:
    """word -> phonemes for the vocabulary seen in the training manifest."""
    from ghana_tts.g2p import english_phonemes, twi_phonemes

    vocab: dict[str, str] = {}
    counts: dict[tuple[str, str], int] = {}
    with open(manifest, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"):
            lang = r.get("language", "twi")
            for w in re.findall(r"[^\W\d_]+", (r.get("text") or "").lower()):
                counts[(lang, w)] = counts.get((lang, w), 0) + 1

    # Most frequent first, so a truncated lexicon still covers most running text.
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    if max_words:
        ranked = ranked[:max_words]
    print(f"lexicon vocabulary: {len(ranked)} (lang, word) pairs", flush=True)

    def one(item):
        (lang, w), _ = item
        try:
            units = (twi_phonemes(w, dialect) if lang == "twi"
                     else english_phonemes(w, symbols))
        except Exception:
            return None
        units = [u for u in units if u in symbols]
        return (w, units) if units else None

    with ThreadPoolExecutor(max_workers=threads) as ex:
        for res in ex.map(one, ranked):
            if res:
                w, units = res
                vocab.setdefault(w, " ".join(units))

    with open(out, "w", encoding="utf-8") as fh:
        for w in sorted(vocab):
            fh.write(f"{w} {vocab[w]}\n")
    print(f"lexicon.txt: {len(vocab)} words")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-config", required=True,
                    help="config.json written during training")
    ap.add_argument("--manifest", required=True, help="training manifest.tsv")
    ap.add_argument("--out", required=True, help="voice directory to create")
    ap.add_argument("--piper-python", default=sys.executable,
                    help="interpreter with piper.train installed")
    ap.add_argument("--top-n", type=int, default=10, help="voices to expose per language")
    ap.add_argument("--min-hours", type=float, default=1.0,
                    help="skip voices with less training audio than this")
    ap.add_argument("--dialect", default="Asante Twi")
    ap.add_argument("--lexicon", action="store_true",
                    help="also build lexicon.txt for native runtimes")
    ap.add_argument("--lexicon-max-words", type=int, default=80_000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--skip-onnx", action="store_true", help="reuse an existing model.onnx")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.train_config).read_text(encoding="utf-8"))
    id_map = cfg["phoneme_id_map"]
    spk_map = cfg.get("speaker_id_map") or {}

    if not args.skip_onnx:
        export_onnx(Path(args.checkpoint), out / "model.onnx", args.piper_python)

    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    write_tokens(id_map, out / "tokens.txt")

    voices = build_registry(args.manifest, spk_map, top_n=args.top_n,
                            min_hours=args.min_hours)
    (out / "voices.json").write_text(json.dumps(
        {"voices": [v.__dict__ for v in voices],
         "total_speakers_in_checkpoint": cfg.get("num_speakers", len(spk_map)),
         "note": ("Curated shortlist ranked by training hours. The checkpoint contains many "
                  "more pseudo-speakers, most with too little audio to sound stable; they "
                  "remain reachable by raw speaker index.")},
        ensure_ascii=False, indent=1))
    print(f"voices.json: {len(voices)} voices "
          f"of {cfg.get('num_speakers')} in the checkpoint")
    for v in voices:
        print(f"  {v}")

    if args.lexicon:
        build_lexicon(Path(args.manifest), set(id_map), out / "lexicon.txt",
                      args.dialect, args.threads, args.lexicon_max_words)

    print(f"\nvoice directory ready: {out}")


if __name__ == "__main__":
    main()

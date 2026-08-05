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
from stable_twi_tts.voices import build_registry  # noqa: E402


def export_onnx(checkpoint: Path, out: Path, piper_python: str) -> None:
    print(f"exporting {checkpoint.name} -> {out.name}", flush=True)
    subprocess.run([piper_python, "-m", "piper.train.export_onnx",
                    "--checkpoint", str(checkpoint), "--output-file", str(out)],
                   check=True)


def write_onnx_metadata(path: Path, cfg: dict, n_speakers: int) -> None:
    """Stamp the metadata a native runtime reads out of the graph.

    Piper's exporter leaves `metadata_props` empty, and sherpa-onnx refuses to load a VITS model
    without it — `'sample_rate' does not exist in the metadata`. Two of these fields matter more
    than they look:

      add_blank   1 tells the runtime to interleave the blank symbol between phonemes, which is
                  exactly Piper's training convention (BOS PAD (phoneme PAD)* EOS). Without it a
                  native runtime feeds half-length sequences and the duration predictor is wrong.
      n_speakers  the runtime uses it to validate the speaker id, so a multi-speaker model
                  declared as single-speaker silently ignores the id and always speaks as
                  speaker 0.
    """
    import onnx

    m = onnx.load(str(path))
    while len(m.metadata_props):
        m.metadata_props.pop()
    meta = {
        "model_type": "vits",
        "comment": "piper",
        "sample_rate": cfg.get("audio", {}).get("sample_rate", 22050),
        "add_blank": 1,
        "n_speakers": n_speakers,
        "language": "twi",
        "voice": cfg.get("espeak", {}).get("voice", "tw"),
        "has_espeak": 0,
        "phoneme_type": cfg.get("phoneme_type", "text"),
    }
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = str(k), str(v)
    onnx.save(m, str(path))
    print(f"onnx metadata: {meta}")


def write_tokens(id_map: dict[str, list[int]], path: Path) -> None:
    """tokens.txt as '<symbol> <id>' per line, ordered by id.

    The space symbol is written as a **literal space**, giving a line that begins with one:

        " 3"

    That looks malformed and is not. sherpa-onnx splits each line on its last whitespace, so a
    leading space parses as the symbol. Substituting a placeholder like `<space>` instead makes
    sherpa throw `IndexError: _Map_base::at` on every single utterance, because it looks up " "
    to separate words and the map has no such key — a failure whose message points nowhere near
    the cause.
    """
    rows = sorted(((v[0], k) for k, v in id_map.items()), key=lambda r: r[0])
    with open(path, "w", encoding="utf-8") as fh:
        for idx, sym in rows:
            fh.write(f"{sym} {idx}\n")
    print(f"tokens.txt: {len(rows)} symbols")


# sherpa-onnx's word splitter breaks on any character that is not an ASCII letter, which
# includes the Twi vowels ɔ and ɛ: `onyankopɔn` is split at the ɔ and the fragment is reported
# OOV and dropped. So lexicon keys are written in ASCII with these two substitutions, and a
# native caller applies the same two replacements to its input text.
#
# q and x are the stand-ins because **Akan has no q or x**, so they cannot collide with a real
# Twi word. Measured over the 78k-word lexicon: 3 collisions, all with English words such as
# `acquire`. The obvious-looking alternatives are far worse -- ɔ→ooo / ɛ→eee produces 49
# collisions between genuine Twi minimal pairs (`aseɛ` and `asɛe` both become `aseeee`), because
# it clashes with real Twi vowel sequences.
ASCII_SUB = {"ɔ": "q", "ɛ": "x"}


def asciify(word: str) -> str:
    for src, dst in ASCII_SUB.items():
        word = word.replace(src, dst)
    return word


def build_lexicon(manifest: Path, symbols: set[str], out: Path, dialect: str,
                  threads: int, max_words: int) -> None:
    """word -> phonemes for the vocabulary seen in the training manifest.

    Written twice: `lexicon.txt` with the true Twi spellings, and `lexicon_ascii.txt` with keys
    asciified for sherpa-onnx. Native runtimes need the second one; anything that can hold a
    Unicode key should use the first.
    """
    from stable_twi_tts.g2p import english_phonemes, twi_phonemes

    vocab: dict[str, str] = {}
    counts: dict[tuple[str, str], int] = {}
    with open(manifest, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"):
            lang = r.get("language", "twi")
            for w in re.findall(r"[^\W\d_]+", (r.get("text") or "").lower()):
                counts[(lang, w)] = counts.get((lang, w), 0) + 1

    # A lexicon is keyed by spelling alone, but the same spelling occurs in both languages with
    # different pronunciations — `a`, `no`, `me`, `aa` are all common Twi words *and* English
    # ones. Twi wins every collision: this is a Twi voice, Twi homographs are far more frequent
    # in Twi text, and handing them English phonemes (with stress and length marks they should
    # never carry) is much the worse failure. English keeps only spellings Twi never uses.
    twi_words = {w for (lang, w) in counts if lang == "twi"}
    shadowed = sum(1 for (lang, w) in counts if lang != "twi" and w in twi_words)
    # Most frequent first, so a truncated lexicon still covers most running text.
    ranked = [kv for kv in sorted(counts.items(), key=lambda kv: -kv[1])
              if kv[0][0] == "twi" or kv[0][1] not in twi_words]
    if max_words:
        ranked = ranked[:max_words]
    print(f"lexicon vocabulary: {len(ranked)} (lang, word) pairs "
          f"({shadowed} English spellings deferred to their Twi homograph)", flush=True)

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
    print(f"{out.name}: {len(vocab)} words")

    # The sherpa-compatible copy. Collisions are reported rather than hidden: an asciified key
    # that already exists means one word's pronunciation would overwrite another's.
    ascii_out = out.with_name(out.stem + "_ascii" + out.suffix)
    seen: dict[str, str] = {}
    collisions = []
    with open(ascii_out, "w", encoding="utf-8") as fh:
        for w in sorted(vocab):
            k = asciify(w)
            if k in seen and seen[k] != w:
                collisions.append((seen[k], w, k))
                continue
            seen[k] = w
            fh.write(f"{k} {vocab[w]}\n")
    print(f"{ascii_out.name}: {len(seen)} words, {len(collisions)} collisions dropped"
          + (f" (e.g. {collisions[0][0]!r} vs {collisions[0][1]!r})" if collisions else ""))


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
    ap.add_argument("--lexicon-max-words", type=int, default=0,
                   help="0 = the whole manifest vocabulary; the lexicon is the only way a "
                        "native runtime can pronounce Twi, so truncating it silently drops words")
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
    if (out / "model.onnx").exists():
        write_onnx_metadata(out / "model.onnx", cfg, cfg.get("num_speakers", len(spk_map)))

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

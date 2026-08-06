"""Precompute an English pronunciation lexicon, so a device needs no espeak at runtime.

espeak's English is 7,132 context-sensitive rules plus 5,794 exceptions, so it cannot be ported
the way Twi's 42-rule table can. But it only has to run *once*: phonemise a large word list here,
ship the result as data, and the device does a dictionary lookup instead of linking a library.

The trade is vocabulary. `libespeak-ng` is 1.56 MB and pronounces anything, including words
invented yesterday; a lexicon is a few MB and silently drops whatever it does not contain. Prefer
the library unless you genuinely cannot link native code — a Flutter web target, a sandboxed
runtime, a pure-JS deployment.

Coverage is reported against real text rather than asserted, because "126,000 words" sounds
complete and is not: proper nouns, inflections and numerals are exactly what a fixed list misses.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORD = re.compile(r"[a-z']+")


def espeak_batch(words: list[str], voice: str, threads: int) -> dict[str, str]:
    """Phonemise each word on its own.

    One espeak call per word rather than one per line: espeak applies sentence-level rules —
    stress placement, function-word reduction — that would make a word's phonemes depend on its
    neighbours in the batch, which is wrong for a lexicon consulted per word.
    """
    def one(w: str) -> tuple[str, str]:
        try:
            r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, "--", w],
                               capture_output=True, text=True, check=True, timeout=10)
            return w, " ".join(r.stdout.split())
        except Exception:
            return w, ""

    out: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for i, (w, ipa) in enumerate(ex.map(one, words), 1):
            if ipa:
                out[w] = ipa
            if i % 20000 == 0:
                print(f"  {i}/{len(words)}", flush=True)
    return out


def tokenize(ipa: str, symbols) -> list[str]:
    """Same greedy longest-match the runtime uses; whitespace skipped, never emitted."""
    order = sorted([s for s in symbols if s.strip()], key=len, reverse=True)
    out, i = [], 0
    while i < len(ipa):
        if ipa[i].isspace():
            i += 1
            continue
        for s in order:
            if ipa.startswith(s, i):
                out.append(s)
                i += len(s)
                break
        else:
            i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="model config.json, for the symbol set")
    ap.add_argument("--out", default="english_lexicon.json")
    ap.add_argument("--voice", default="en-us")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--extra-words", default=None,
                    help="additional word list, one per line — add your domain vocabulary")
    ap.add_argument("--coverage-text", default=None,
                    help="a sample of real text, to measure token coverage honestly")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    symbols = set(json.loads(Path(args.config).read_text(encoding="utf-8"))["phoneme_id_map"])

    try:
        import cmudict
        words = sorted(cmudict.dict())
    except ImportError:
        print("needs the cmudict word list: pip install cmudict", file=sys.stderr)
        raise SystemExit(1)

    if args.extra_words:
        extra = [w.strip().lower() for w in
                 Path(args.extra_words).read_text(encoding="utf-8").splitlines()
                 if w.strip()]
        words = sorted(set(words) | set(extra))
        print(f"added {len(extra)} extra words")

    words = [w for w in words if WORD.fullmatch(w)]
    if args.limit:
        words = words[: args.limit]
    print(f"phonemising {len(words)} words with espeak-ng {args.voice}")

    raw = espeak_batch(words, args.voice, args.threads)

    lex: dict[str, str] = {}
    unmapped: set[str] = set()
    for w, ipa in raw.items():
        units = tokenize(ipa, symbols)
        missing = [u for u in units if u not in symbols]
        if missing:
            unmapped.update(missing)
            continue
        if units:
            lex[w] = " ".join(units)

    Path(args.out).write_text(json.dumps({
        "note": ("English word -> phoneme units, precomputed with espeak-ng so a device needs no "
                 "espeak at runtime. Units are already tokenised against the model's symbol set: "
                 "split on spaces, do not re-tokenise. Words absent here cannot be pronounced — "
                 "link libespeak-ng (1.56 MB) if you need unlimited vocabulary."),
        "voice": args.voice,
        "n_words": len(lex),
        "lexicon": lex,
    }, ensure_ascii=False), encoding="utf-8")

    size = Path(args.out).stat().st_size / 1e6
    print(f"\n{len(lex)} words -> {args.out} ({size:.1f} MB)")
    if unmapped:
        print(f"  {len(unmapped)} symbols espeak produced that the model lacks: "
              f"{sorted(unmapped)[:12]}")

    if args.coverage_text:
        text = Path(args.coverage_text).read_text(encoding="utf-8").lower()
        toks = WORD.findall(text)
        hit = sum(1 for t in toks if t in lex)
        types = set(toks)
        thit = sum(1 for t in types if t in lex)
        print(f"\ncoverage on {Path(args.coverage_text).name}:")
        print(f"  tokens: {hit}/{len(toks)} ({100*hit/max(len(toks),1):.1f}%) "
              f"— what a listener actually hears")
        print(f"  types:  {thit}/{len(types)} ({100*thit/max(len(types),1):.1f}%)")
        miss = [t for t in types if t not in lex][:15]
        if miss:
            print(f"  missed e.g.: {miss}")


if __name__ == "__main__":
    main()

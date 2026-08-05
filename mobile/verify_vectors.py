"""Independent reimplementation of the front-end, checked against the test vectors.

This exists to validate the *algorithm* the Kotlin and Swift ports transcribe — it deliberately
imports nothing from `stable_twi_tts`, reading only `twi_rules.json`, the model's `config.json`
and espeak, exactly as a native port does. If this passes, the algorithm as documented is correct
and any faithful port of it will pass too.

Run it after changing a port, or as a template for a port's own test.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

PAD, BOS, EOS = "_", "^", "$"
SPAN = re.compile(r"\[([^\]]*)\]")


def twi_phonemes(text: str, graphemes: dict[str, str], symbols=()) -> list[str]:
    """Twi text -> phoneme units.

    Characters with no grapheme rule are passed through when the model has a symbol for them.
    That is what carries punctuation: the model was trained with `.` `,` `?` as units, so
    dropping them changes the prosody it was taught.
    """
    keys = sorted(graphemes, key=len, reverse=True)   # longest first
    out, i, w = [], 0, text.lower()
    while i < len(w):
        for k in keys:
            if w.startswith(k, i):
                out.append(graphemes[k])
                i += len(k)
                break
        else:
            # Punctuation yes, whitespace no: the model has a space symbol but was never
            # trained with it, so emitting it feeds sequences unlike anything seen.
            if not w[i].isspace() and w[i] in symbols:
                out.append(w[i])
            i += 1
    return out


def tokenize(ipa: str, symbols) -> list[str]:
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


def espeak_ipa(text: str, voice: str = "en-us") -> str:
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, "--", text],
                       capture_output=True, text=True, check=True)
    return " ".join(r.stdout.split())


def to_ids(units: list[str], id_map: dict, language: str) -> list[int]:
    ids = list(id_map[BOS]) + list(id_map[PAD])
    tok = f"«{'twi' if language == 'mixed' else language}»"
    if tok in id_map:
        ids += list(id_map[tok]) + list(id_map[PAD])
    for u in units:
        ids += list(id_map[u]) + list(id_map[PAD])
    return ids + list(id_map[EOS])


def units_for(case: dict, graphemes: dict, symbols) -> list[str]:
    text, lang = case["text"], case["language"]
    if lang == "twi":
        return twi_phonemes(text, graphemes, symbols)
    if lang == "eng":
        return tokenize(espeak_ipa(text), symbols)
    # mixed: [bracketed] spans are English inside a Twi frame
    units, pos = [], 0
    for m in SPAN.finditer(text):
        head = text[pos:m.start()].strip()
        if head:
            units += twi_phonemes(head, graphemes, symbols)
        span = m.group(1).strip()
        if span:
            units += tokenize(espeak_ipa(span), symbols)
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        units += twi_phonemes(tail, graphemes, symbols)
    return units


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="directory holding the json files")
    ap.add_argument("--config", required=True, help="model config.json")
    args = ap.parse_args()

    d = Path(args.dir)
    graphemes = json.loads((d / "twi_rules.json").read_text(encoding="utf-8"))["graphemes"]
    vectors = json.loads((d / "test_vectors.json").read_text(encoding="utf-8"))
    id_map = json.loads(Path(args.config).read_text(encoding="utf-8"))["phoneme_id_map"]
    symbols = set(id_map)

    passed = failed = skipped = 0
    for case in vectors["cases"]:
        try:
            units = units_for(case, graphemes, symbols)
        except FileNotFoundError:
            skipped += 1
            print(f"  SKIP  {case['language']:5} (espeak-ng not installed)  {case['text'][:40]}")
            continue
        ids = to_ids(units, id_map, case["language"]) if units else []
        ok_u = units == case["units"]
        ok_i = ids == case["ids"]
        if ok_u and ok_i:
            passed += 1
            print(f"  PASS  {case['language']:5} {len(units):3} units  {case['text'][:44]}")
        else:
            failed += 1
            print(f"  FAIL  {case['language']:5} {case['text'][:44]}")
            if not ok_u:
                print(f"          want units: {' '.join(case['units'][:18])}")
                print(f"          got  units: {' '.join(units[:18])}")
            elif not ok_i:
                print(f"          ids differ at "
                      f"{next(i for i, (a, b) in enumerate(zip(case['ids'], ids)) if a != b)}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

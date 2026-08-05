"""Generate test vectors so a native port can be *verified*, not hoped-for.

Every step of the front-end has a failure mode that produces fluent-sounding wrong audio rather
than an error: a dropped pad halves the sequence, a split multi-character symbol becomes an
unknown unit, an emitted space token feeds the model something it never trained on. None of those
crash. So a port needs ground truth at each stage, which is what this writes.

Also emits `twi_rules.json` — the 42-entry Twi table as data, so a port carries no transcription
of it in source.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TWI = [
    "Akwaaba, wo ho te sɛn?",
    "Me da wo ase paa.",
    "Onyankopɔn adɔeɛ",
    "Mepɛ sɛ mesua Twi kasa yie.",
    "Ɛho nhwɛsoɔ bi nie.",
    "Yɛfrɛ me Kwame na mefiri Kumasi.",
    "nkyekyɛm nwɔtwe",          # digraph stress test: nky, kyɛ, nwɔ, tw
    "aa ee ii oo",              # long vowels
]
ENG = [
    "Good morning, welcome to the news.",
    "I think this is the third one.",
    "Accra is the capital city of Ghana.",
]
MIXED = [
    "Mepɛ sɛ mesua [computer science] wɔ [University of Ghana].",
    "Ɔkyerɛkyerɛni no kaa sɛ [the examination] bɛba [next week].",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="voice directory")
    ap.add_argument("--out", default="mobile")
    ap.add_argument("--twi-rules", required=True,
                    help="africa-g2p languages/twi.json")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from stable_twi_tts.g2p import phonemize, to_ids

    cfg = json.loads((Path(args.model) / "config.json").read_text(encoding="utf-8"))
    id_map = cfg["phoneme_id_map"]
    symbols = set(id_map)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- the Twi table, as data ----
    spec = json.loads(Path(args.twi_rules).read_text(encoding="utf-8"))
    graphemes = spec["graphemes"]
    (out / "twi_rules.json").write_text(json.dumps({
        "note": ("Longest-match, context-free. Sort keys by descending length before matching, "
                 "or 'ky' is read as 'k' + 'y'. Lowercase the input first."),
        "source": spec.get("source"),
        "graphemes": graphemes,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    cases = []
    for lang, texts in (("twi", TWI), ("eng", ENG), ("mixed", MIXED)):
        for text in texts:
            try:
                units = phonemize(text, lang, symbols)
                ids = to_ids(units, id_map, lang)
            except Exception as e:
                print(f"  skipped {text!r}: {type(e).__name__}: {e}")
                continue
            cases.append({"language": lang, "text": text,
                          "units": units, "n_units": len(units), "ids": ids})
            print(f"  {lang:5} {len(units):3} units {len(ids):4} ids  {text[:48]}")

    (out / "test_vectors.json").write_text(json.dumps({
        "note": ("Ground truth from the Python front-end. A port passes when `units` and `ids` "
                 "match exactly for every case. Twi needs only twi_rules.json; English cases "
                 "need espeak-ng with the en-us voice."),
        "model_symbols": len(id_map),
        "conventions": {
            "wrap": "ids = [BOS] + [PAD] + (id(unit) + [PAD] for each unit) + [EOS]",
            "language_token": "inserted after the leading PAD when present in the map",
            "whitespace": "skipped when tokenising; never emitted as the space symbol",
            "match": "greedy longest-first over the symbol set",
            "case": "Twi input is lowercased; English is passed to espeak unchanged",
        },
        "special_ids": {k: id_map[k] for k in ("_", "^", "$", " ") if k in id_map},
        "language_tokens": {k: v for k, v in id_map.items()
                            if k.startswith("«") and k.endswith("»")},
        "cases": cases,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{len(cases)} cases -> {out / 'test_vectors.json'}")
    print(f"{len(graphemes)} Twi rules -> {out / 'twi_rules.json'}")


if __name__ == "__main__":
    main()

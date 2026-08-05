"""Rank voices by measured intelligibility instead of by training hours.

Hours is a proxy, and it turns out to be a poor one. The English voice with the most audio
(13.1 h) sounds clearly worse than the top Twi voice (11.2 h), because the model learned English
less well overall — so a ranking by hours actively misleads, presenting a weak voice as the best
choice for its language.

This measures each voice directly: synthesise the same held-out utterances with every voice, run
the phoneme recogniser over the output, and score unit error against the phonemes that were
requested. Same text for every voice, so the only variable is the voice.

What it measures is articulation, not pleasantness. A voice that scores well is intelligible; a
listener still has to decide whether it is nice. Treat this as a floor filter — it reliably
identifies voices that are *bad* — rather than as a ranking of beauty.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path


def edit_distance(a: list, b: list) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def is_punct(u: str) -> bool:
    return len(u) == 1 and unicodedata.category(u).startswith("P")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-dir", required=True)
    ap.add_argument("--manifest", required=True,
                    help="held-out TSV with id, ipa, language")
    ap.add_argument("--synth-script", required=True,
                    help="synth_piper.py, for generating with a chosen speaker")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--asr-model", required=True, help="Ghana phoneme ASR release dir")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--per-voice", type=int, default=20, help="utterances per voice")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", default=None, help="write the updated voices.json here")
    args = ap.parse_args()

    vd = Path(args.voice_dir)
    voices = json.loads((vd / "voices.json").read_text(encoding="utf-8"))
    work = Path(args.workdir or tempfile.mkdtemp(prefix="rankvoices-"))
    work.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE,
                                   escapechar="\\"))

    sys.path.insert(0, str(Path(args.asr_model).parent))
    scored = []
    for v in voices["voices"]:
        lang, name, src = v["language"], v["name"], v["source_speaker"]
        # Same text for every voice within a language, so voices are directly comparable.
        pool = [r for r in rows if r.get("language") == lang][: args.per_voice]
        if not pool:
            scored.append({**v, "measured_uer": None})
            continue

        sub = work / f"{name}.tsv"
        with open(sub, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t",
                               quoting=csv.QUOTE_NONE, escapechar="\\")
            w.writeheader()
            for r in pool:
                w.writerow({**r, "speaker": src})

        outdir = work / f"wav_{name}"
        subprocess.run([args.python, args.synth_script, "--checkpoint", args.checkpoint,
                        "--config", args.config, "--manifest", str(sub),
                        "--out", str(outdir), "--limit", str(len(pool)),
                        "--speaker", src],
                       check=True, capture_output=True)

        from phonemise import MODEL_SR, ctc_collapse, load_model, make_batches, run_batch
        import soundfile as sf

        model, keep, vocab = load_model(Path(args.asr_model), "cuda")
        wavs, refs = [], []
        for r in pool:
            p = outdir / f"{r['id']}.wav"
            if not p.exists():
                continue
            wav, sr = sf.read(p, dtype="float32")
            wavs.append((wav, int(sr)))
            refs.append([u for u in r["ipa"].split(" ") if u and not is_punct(u)])

        if not wavs:
            scored.append({**v, "measured_uer": None})
            continue

        lens = [round(len(w) * MODEL_SR / sr) for w, sr in wavs]
        hyps: list = [None] * len(wavs)
        cache: dict = {}
        for b in make_batches(lens, list(range(len(wavs))), MODEL_SR * 120, 8):
            res = run_batch(model, keep, [wavs[i][0] for i in b],
                            [wavs[i][1] for i in b], cache, "cuda")
            for i, ids in zip(b, res):
                hyps[i] = [u for u in ctc_collapse(ids, vocab) if not is_punct(u)]

        e = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
        n = sum(len(r) for r in refs)
        uer = round(e / max(n, 1), 4)
        scored.append({**v, "measured_uer": uer})
        print(f"  {name:8} {lang:4} {v['hours']:6.2f} h  measured UER {uer:.2%}", flush=True)

    # Rank within language by measured error, so the exposed order reflects what a listener
    # would actually prefer rather than how much audio happened to exist.
    ranked = sorted(scored, key=lambda v: (v["language"],
                                           v["measured_uer"] if v["measured_uer"] is not None
                                           else 9.0))
    for lang in sorted({v["language"] for v in ranked}):
        for i, v in enumerate([x for x in ranked if x["language"] == lang], 1):
            v["name"] = f"{lang}-{i}"

    voices["voices"] = ranked
    voices["ranking"] = ("measured round-trip phoneme error on held-out text, lower is better; "
                         "hours retained for reference")
    dest = Path(args.out or vd / "voices.json")
    dest.write_text(json.dumps(voices, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    for v in ranked:
        u = v["measured_uer"]
        print(f"  {v['name']:8} {v['language']:4} {v['hours']:6.2f} h  "
              f"{'UER ' + format(u, '.2%') if u is not None else 'unmeasured'}  "
              f"{v['source_speaker']}")


if __name__ == "__main__":
    main()

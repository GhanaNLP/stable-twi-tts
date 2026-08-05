"""Assemble everything needed to continue training this model, and nothing else.

A published .onnx lets people *use* a voice. Continuing training needs the Lightning checkpoint
plus the exact inputs that produced it — the phoneme id map, the speaker id map, and the training
invocation. Publishing the checkpoint without those is a common and frustrating omission: the
weights load, and then the phoneme ids mean something different and the output is nonsense.

Audio is deliberately not bundled. It is ~25 GB and already published as a Hugging Face dataset,
so the bundle records how to regenerate the manifests from it instead of duplicating it.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

NOTES = """# Continuing training from this checkpoint

## What is here

| file | what it is |
|---|---|
| `checkpoint.ckpt` | PyTorch Lightning checkpoint (generator, discriminators, optimiser state) |
| `config.json` | sample rate, phoneme id map, speaker id map — **required to interpret the weights** |
| `phonemes.json` | symbol -> id map, the same one used to build the training targets |
| `train_command.sh` | the exact invocation that produced this checkpoint |
| `dataset_stats.json` | what it was trained on |

**The phoneme id map is not optional.** The weights encode "id 26 means /n/". Load them against a
different map and the model reads a different language. If you extend the inventory, keep every
existing id fixed and add new symbols in the free slots — ids up to 255 are available and {n_used}
are used.

## Getting the audio

The audio is not in this bundle (~25 GB). It is published as a dataset:

    huggingface-cli download {dataset} --repo-type dataset --local-dir data/

Rebuild the wav directory and manifest with `tts_data.py` from
https://github.com/GhanaNLP/phoneme-asr-batch, then regenerate the Piper metadata. The pipeline
is deterministic, so you get the same targets.

## Adding a language or new speakers

Two things bit us, and will bite you:

1. **Resize the speaker table before loading.** `model_g.emb_g.weight` has one row per speaker.
   A different speaker count fails to load; `adapt_piper_ckpt.py` resizes it and seeds new rows
   from real pretrained voices rather than noise, which converges much faster than random init.
2. **Clear the phoneme cache.** Piper caches phoneme tensors keyed by *text*, not by phoneme id.
   Change the id map and the stale tensors are silently reused, so you train on the old targets
   while believing you changed them. Delete `cache/*.phonemes.pt` and keep `*.audio.pt`.

## What we learned, so you do not repeat it

**Training targets and synthesis input must come from the same function.** We trained on
ASR-derived phonemes and synthesised from G2P-derived ones. They disagreed on 26% of units for
Twi and 51% for English, and the TTS phoneme error tracked that almost exactly — 25.6% vs 68.6%.
Nothing errored; it just sounded wrong. Use the same phonemiser on both sides.

**Do not early-stop on `val_mel`.** It saturates while the adversarial losses are still removing
artifacts. We kept top-5 by `val_mel` and top-5 by `val_mos` (UTMOS), and separately measured a
round-trip phoneme error: synthesise held-out phonemes, re-recognise, compare. That last one is
the only metric that tracked what a listener heard.

**Hours is a poor proxy for voice quality.** Our best-measured Twi voice had half the audio of the
one with the most. Measure voices; do not rank them by data volume.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--phonemes", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--train-script", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="ghanaopendata/new-twi-tts-aligned-ipa")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.checkpoint, out / "checkpoint.ckpt")
    shutil.copy2(args.config, out / "config.json")
    shutil.copy2(args.phonemes, out / "phonemes.json")
    if args.train_script and Path(args.train_script).exists():
        shutil.copy2(args.train_script, out / "train_command.sh")

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    id_map = cfg["phoneme_id_map"]
    n_used = len({i for v in id_map.values() for i in v})

    langs: Counter = Counter()
    secs: Counter = Counter()
    speakers: set = set()
    with open(args.manifest, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"):
            lang = r.get("language", "twi")
            langs[lang] += 1
            secs[lang] += float(r["duration"])
            speakers.add(r["speaker"])

    stats = {
        "sample_rate": cfg.get("audio", {}).get("sample_rate"),
        "num_speakers": cfg.get("num_speakers"),
        "num_symbols": cfg.get("num_symbols"),
        "phoneme_symbols_used": n_used,
        "speakers_in_manifest": len(speakers),
        "by_language": {k: {"clips": langs[k], "hours": round(secs[k] / 3600, 2)}
                        for k in sorted(langs)},
        "audio_dataset": args.dataset,
        "checkpoint": Path(args.checkpoint).name,
    }
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=1))
    (out / "FINETUNING.md").write_text(NOTES.format(n_used=n_used, dataset=args.dataset))

    total = sum(v["hours"] for v in stats["by_language"].values())
    print(f"bundle: {out}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:20} {f.stat().st_size/1e6:8.1f} MB")
    print(f"\n{sum(langs.values())} clips, {total:.1f} h, {len(speakers)} speakers, "
          f"{n_used} phoneme symbols")


if __name__ == "__main__":
    main()

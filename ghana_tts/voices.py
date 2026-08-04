"""The voice registry: a curated shortlist rather than every speaker in the checkpoint.

The model has 1,555 speaker embeddings, but they are pseudo-speakers derived by clustering
x-vectors over unlabelled broadcast audio, and the data behind them is wildly uneven — the
busiest voice has 11 hours, the median has minutes. An embedding trained on a handful of clips
produces an unstable, muddy voice, so exposing all 1,555 as if they were equivalent choices
would mostly offer users ways to get a bad result.

So the registry ships a shortlist ranked by training hours, and `voices.json` records the hours
behind each one. Anything not on the list is still reachable by raw speaker id for anyone who
wants to explore, but it is opt-in rather than the default menu.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Voice:
    name: str            # stable public name, e.g. "twi-1"
    speaker_id: int      # index into the model's speaker embedding table
    language: str        # twi | eng
    hours: float         # training audio behind this voice
    clips: int
    source_speaker: str  # pseudo-speaker id it came from, e.g. spk_0006

    def __str__(self) -> str:
        return f"{self.name} ({self.language}, {self.hours:.1f} h)"


class VoiceRegistry:
    def __init__(self, voices: list[Voice]):
        self._by_name = {v.name: v for v in voices}
        self._by_source = {v.source_speaker: v for v in voices}
        self.voices = voices

    @classmethod
    def load(cls, path: str | Path) -> "VoiceRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Voice(**v) for v in data["voices"]])

    def get(self, name_or_id: str | int) -> Voice:
        """Accepts a public name, a source speaker id, or a raw integer speaker index.

        The raw-integer path is the escape hatch for the 1,535 voices not on the shortlist:
        it works, but you are on your own for quality.
        """
        if isinstance(name_or_id, int):
            return Voice(f"raw-{name_or_id}", name_or_id, "unknown", 0.0, 0, "")
        if name_or_id in self._by_name:
            return self._by_name[name_or_id]
        if name_or_id in self._by_source:
            return self._by_source[name_or_id]
        if str(name_or_id).isdigit():
            return self.get(int(name_or_id))
        raise KeyError(
            f"unknown voice {name_or_id!r}. Available: "
            f"{', '.join(sorted(self._by_name))}"
        )

    def by_language(self, language: str) -> list[Voice]:
        return [v for v in self.voices if v.language == language]

    def default(self, language: str) -> Voice:
        opts = self.by_language(language)
        if not opts:
            raise KeyError(f"no voices for language {language!r}")
        return max(opts, key=lambda v: v.hours)

    def describe(self) -> str:
        lines = [f"{'voice':10} {'lang':5} {'hours':>7}  {'clips':>7}  source"]
        for v in sorted(self.voices, key=lambda v: (v.language, -v.hours)):
            lines.append(f"{v.name:10} {v.language:5} {v.hours:7.2f}  {v.clips:7}  "
                         f"{v.source_speaker}")
        return "\n".join(lines)


def build_registry(manifest: str | Path, speaker_id_map: dict[str, int],
                   top_n: int = 10, min_hours: float = 0.0) -> list[Voice]:
    """Rank speakers by training hours and keep the top N per language.

    Hours, not clip count: the Twi clips average 3.9 s and the English 13.6 s, so ranking by
    clips would systematically favour Twi speakers and misrepresent how much audio is actually
    behind each voice.
    """
    import csv
    from collections import defaultdict

    agg: dict[str, dict] = defaultdict(lambda: {"clips": 0, "secs": 0.0, "language": ""})
    with open(manifest, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"):
            a = agg[r["speaker"]]
            a["clips"] += 1
            a["secs"] += float(r["duration"])
            a["language"] = r.get("language", "twi")

    out: list[Voice] = []
    for lang in sorted({a["language"] for a in agg.values()}):
        ranked = sorted(((k, v) for k, v in agg.items() if v["language"] == lang),
                        key=lambda kv: -kv[1]["secs"])
        n = 0
        for src, a in ranked:
            if src not in speaker_id_map:
                continue  # dropped from training entirely
            hours = a["secs"] / 3600
            if hours < min_hours or n >= top_n:
                break
            n += 1
            out.append(Voice(name=f"{lang}-{n}", speaker_id=speaker_id_map[src],
                             language=lang, hours=round(hours, 2), clips=a["clips"],
                             source_speaker=src))
    return out

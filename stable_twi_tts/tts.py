"""ONNX inference for the bilingual Twi/Ghanaian-English Piper voice.

onnxruntime is the runtime rather than torch: it is a ~50 MB dependency instead of ~2 GB, runs
on CPU at many times realtime, and installs on Linux, macOS, Windows, ARM and in the browser.
The exported .onnx is the same artifact sherpa-onnx consumes, so a native or mobile deployment
can load it without touching this file — see the README for that route and its one limitation.

Long inputs are split on sentence boundaries and concatenated. The training clips were 4-14
seconds, and VITS attention degrades on inputs much longer than anything it saw, so a whole
paragraph in one forward pass tends to slur or truncate near the end.
"""
from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .g2p import phonemize, split_sentences, to_ids
from .voices import Voice, VoiceRegistry


@dataclass
class Synthesis:
    audio: np.ndarray     # float32 in [-1, 1]
    sample_rate: int
    voice: Voice
    n_phonemes: int

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.clip(self.audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm.tobytes())
        return p


class StableTwiTTS:
    """Load a voice directory containing model.onnx, config.json and voices.json."""

    DEFAULT_REPO = "ghanaopendata/stable-twi-tts"

    @classmethod
    def from_pretrained(cls, repo_id: str | None = None, revision: str | None = None,
                        cache_dir: str | Path | None = None, quiet: bool = False,
                        **kwargs) -> "StableTwiTTS":
        """Download the published voice and load it. Needs no extra dependency.

        By default the four inference files come from a GitHub release over `urllib`, with their
        checksums verified — see download.py for why that beats huggingface_hub, which would add
        twelve transitive packages to fetch 77 MB.

        Passing `repo_id` or `revision` selects the Hub instead, and that needs
        `pip install 'stable-twi-tts[hub]'`. Use it to load a fork, or to pin a revision other
        than the release this version ships against. The Hub repo also holds a ~930 MB training
        checkpoint under `finetune/`, so only inference files are requested.
        """
        if repo_id is None and revision is None:
            from .download import ensure_model
            return cls(ensure_model(cache_dir, quiet=quiet), **kwargs)

        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "repo_id and revision select the Hugging Face Hub, which needs:\n"
                "  pip install 'stable-twi-tts[hub]'\n"
                "Call from_pretrained() with no arguments for the GitHub release instead — "
                "same weights, no extra packages.") from e

        path = snapshot_download(
            repo_id or cls.DEFAULT_REPO, revision=revision, cache_dir=cache_dir,
            allow_patterns=["model.onnx", "config.json", "voices.json", "tokens.txt"])
        return cls(path, **kwargs)

    def __init__(self, model_dir: str | Path, providers: list[str] | None = None,
                 num_threads: int | None = None):
        import onnxruntime as ort

        d = Path(model_dir)
        self.dir = d
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"no config.json in {d}")
        self.config = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.id_map: dict[str, list[int]] = self.config["phoneme_id_map"]
        self.symbols = set(self.id_map)
        self.sample_rate = self.config.get("audio", {}).get("sample_rate", 22050)
        self.num_speakers = self.config.get("num_speakers", 1)

        vpath = d / "voices.json"
        self.voices = (VoiceRegistry.load(vpath) if vpath.exists()
                       else VoiceRegistry([]))

        opts = ort.SessionOptions()
        if num_threads:
            opts.intra_op_num_threads = num_threads
        # Model outputs are deterministic given the same seed; disable the spam.
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(d / "model.onnx"), sess_options=opts,
            providers=providers or ["CPUExecutionProvider"])
        self._inputs = {i.name for i in self.session.get_inputs()}

    # -------------------------------------------------------------- inference

    def _forward(self, ids: list[int], sid: int, length_scale: float,
                 noise_scale: float, noise_w: float) -> np.ndarray:
        feed = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array([noise_scale, length_scale, noise_w], dtype=np.float32),
        }
        if "sid" in self._inputs:
            feed["sid"] = np.array([sid], dtype=np.int64)
        out = self.session.run(None, feed)[0]
        return np.asarray(out, dtype=np.float32).squeeze()

    def synthesize(self, text: str, voice: str | int | Voice | None = None,
                   language: str = "twi", length_scale: float = 1.0,
                   noise_scale: float = 0.667, noise_w: float = 0.8,
                   split_long: bool = True, pause: float = 0.20) -> Synthesis:
        """Speak `text`.

        language: "twi", "eng", or "mixed" for code-switched text where [bracketed] spans are
        English. length_scale > 1 slows speech down; noise_scale and noise_w control how much
        prosodic variation the model samples.
        """
        if isinstance(voice, Voice):
            v = voice
        elif voice is None:
            # Code-switched text sits in a Twi frame, so a Twi voice is the right default.
            v = self.voices.default("twi" if language == "mixed" else language)
        else:
            v = self.voices.get(voice)

        chunks = split_sentences(text) if split_long else [text]
        if not chunks:
            raise ValueError("nothing to synthesise")

        pieces, n_ph = [], 0
        gap = np.zeros(int(pause * self.sample_rate), dtype=np.float32)
        for i, chunk in enumerate(chunks):
            units = phonemize(chunk, language, self.symbols)
            if not units:
                continue
            ids = to_ids(units, self.id_map, language)
            n_ph += len(units)
            pieces.append(self._forward(ids, v.speaker_id, length_scale, noise_scale,
                                        noise_w))
            if i < len(chunks) - 1:
                pieces.append(gap)
        if not pieces:
            raise ValueError(f"no pronounceable phonemes in {text!r}")

        return Synthesis(np.concatenate(pieces), self.sample_rate, v, n_ph)

    # ------------------------------------------------------------------ batch

    def synthesize_batch(self, items, outdir: str | Path, workers: int = 4,
                         skip_existing: bool = True, on_done=None, **kw):
        """Synthesise many utterances, writing <id>.wav and returning per-item results.

        onnxruntime releases the GIL during inference, so threads genuinely parallelise here.
        Failures are captured per item rather than aborting the run — one unpronounceable line
        in a 10,000-line corpus should not lose the other 9,999.
        """
        from concurrent.futures import ThreadPoolExecutor

        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        items = list(items)

        def one(it: dict) -> dict:
            uid = str(it["id"])
            dest = out / f"{uid}.wav"
            if skip_existing and dest.exists():
                return {"id": uid, "status": "skipped", "path": str(dest)}
            try:
                merged = {**kw, **{k: v for k, v in it.items()
                                   if k in ("voice", "language", "length_scale",
                                            "noise_scale", "noise_w")}}
                s = self.synthesize(it["text"], **merged)
                s.save(dest)
                return {"id": uid, "status": "ok", "path": str(dest),
                        "duration": round(s.duration, 3), "voice": s.voice.name,
                        "n_phonemes": s.n_phonemes}
            except Exception as e:
                return {"id": uid, "status": "error", "error": f"{type(e).__name__}: {e}"}

        results = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(one, items):
                results.append(r)
                if on_done:
                    on_done(r, len(results), len(items))
        return results

"""Text to phoneme ids, matching exactly how the model was trained.

The model does not read letters. It reads phoneme ids, and it only ever learned the phonemes
its training targets contained. So the single most important property of this file is that it
reproduces the training front-end exactly:

    Twi     ->  ghana-g2p (GhanaNLP)      the same library that produced the Twi targets
    English ->  espeak-ng (en-us)          the same phonemiser that produced the English targets

Substituting a different English G2P is the one change most likely to quietly ruin output. An
earlier version of this pipeline folded English into the Ghanaian inventory (θ→t, æ→a). It
produced valid-looking IPA, disagreed with the training targets on 51% of units, and the model
scored 68.6% phoneme error against Twi's 25.6%. Same-function-both-sides is not a style
preference; it is the difference between working and not.

Code-switching is supported through [bracketed] spans, which are phonemised as English inside a
Twi frame. The model was never trained on code-switched utterances, so this is extrapolation —
it works because both languages draw on one shared inventory and the model consumes phonemes
rather than spelling, so it never has to decide which language a word belongs to.
"""
from __future__ import annotations

import re
import subprocess
from functools import lru_cache

PAD, BOS, EOS = "_", "^", "$"
SPAN_RE = re.compile(r"\[([^\]]*)\]")
SENT_RE = re.compile(r"(?<=[.!?])\s+")


class PhonemeError(RuntimeError):
    pass


def tokenize(ipa: str, symbols) -> list[str]:
    """Greedy longest-match against the model's symbol set.

    Longest-first matters: several symbols are multi-character (the English diphthongs aɪ aʊ
    ɔɪ eɪ oʊ, and Twi's kʰ, t͡ʃ, k͡p, hʷ). Splitting by codepoint would shatter them into
    units the model has never seen.

    Whitespace is skipped rather than emitted, even though the model does have a symbol for a
    space. The English training targets were tokenised with the space symbol excluded, so
    emitting word boundaries here would feed the model sequences unlike anything it trained
    on. It also cannot survive serialisation: a space-separated phoneme string has no way to
    represent a phoneme that *is* a space.
    """
    order = sorted([s for s in symbols if s.strip()], key=len, reverse=True)
    out: list[str] = []
    i = 0
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
            i += 1  # a codepoint the model has no symbol for; dropping beats guessing
    return out


@lru_cache(maxsize=4)
def _twi_engine(dialect: str):
    try:
        from ghana_g2p import GhanaG2P
    except ImportError as e:  # pragma: no cover
        raise PhonemeError(
            "Twi needs ghana-g2p, which is not on PyPI. Install it directly:\n"
            "  pip install git+https://github.com/AfriSpeech/africa-g2p \\\n"
            "              git+https://github.com/GhanaNLP/ghana-g2p\n"
            "or from source (africa-g2p's wheel build is currently broken):\n"
            "  git clone https://github.com/AfriSpeech/africa-g2p\n"
            "  git clone https://github.com/GhanaNLP/ghana-g2p\n"
            "  export PYTHONPATH=africa-g2p/src:ghana-g2p/src"
        ) from e
    return GhanaG2P(dialect)


def twi_phonemes(text: str, dialect: str = "Asante Twi") -> list[str]:
    g = _twi_engine(dialect)
    return [u for u in g.ipa(text, sep=" ", punctuation=True).split(" ") if u]


def english_phonemes(text: str, symbols, voice: str = "en-us") -> list[str]:
    try:
        r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, "--", text],
                           capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise PhonemeError(
            "English needs espeak-ng on PATH. Install with:\n"
            "  apt install espeak-ng     # or: brew install espeak-ng"
        ) from e
    return tokenize(" ".join(r.stdout.split()), symbols)


def mixed_phonemes(text: str, symbols, dialect: str = "Asante Twi") -> list[str]:
    """[bracketed] spans as English, the rest as Twi."""
    units: list[str] = []
    pos = 0
    for m in SPAN_RE.finditer(text):
        head = text[pos:m.start()].strip()
        if head:
            units += twi_phonemes(head, dialect)
        span = m.group(1).strip()
        if span:
            units += english_phonemes(span, symbols)
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        units += twi_phonemes(tail, dialect)
    return units


def phonemize(text: str, language: str, symbols, dialect: str = "Asante Twi") -> list[str]:
    """Text to phoneme units.

    English words are pronounced as English, using the model's own English phonemes, which is
    what it was trained on. Twi uses ghana-g2p and English uses espeak-ng — the same two
    phonemisers that produced the training targets.
    """
    if language == "twi":
        return (mixed_phonemes(text, symbols, dialect) if SPAN_RE.search(text)
                else twi_phonemes(text, dialect))
    if language == "eng":
        return english_phonemes(text, symbols)
    if language == "mixed":
        return mixed_phonemes(text, symbols, dialect)
    raise PhonemeError(f"unknown language {language!r}; use twi, eng or mixed")


def to_ids(units: list[str], id_map: dict, language: str) -> list[int]:
    """Wrap and interleave exactly as training did: BOS PAD (unit PAD)* EOS.

    The pad between every unit is not decoration — Piper's VITS was trained with it, and
    omitting it halves the input length the duration predictor expects.
    """
    missing = sorted({u for u in units if u not in id_map})
    if missing:
        raise PhonemeError(f"the model has no symbol for: {missing}")

    ids = list(id_map[BOS]) + list(id_map[PAD])
    # Per-utterance language token, present only in bilingual models. Mixed text uses the
    # Twi token because Twi is the matrix language in Ghanaian code-switching.
    tok = f"«{'twi' if language == 'mixed' else language}»"
    if tok in id_map:
        ids += list(id_map[tok]) + list(id_map[PAD])
    for u in units:
        ids += list(id_map[u]) + list(id_map[PAD])
    return ids + list(id_map[EOS])


def split_sentences(text: str, max_units: int = 320) -> list[str]:
    """Split long input on sentence boundaries.

    Attention degrades on inputs far longer than anything in training, and the training clips
    were 4-14 seconds. Long paragraphs are better synthesised per sentence and concatenated.
    """
    parts = [p.strip() for p in SENT_RE.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])

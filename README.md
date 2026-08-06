# stable-twi-tts

Speech synthesis for **Twi** and **Ghanaian English**, running on ONNX — a ~50 MB dependency
instead of a ~2 GB PyTorch install, many times realtime on a laptop CPU, and portable to Linux,
macOS, Windows, ARM and mobile.

Two things it does that a generic TTS wrapper doesn't:

- **Voices are ranked by measured intelligibility**, not by how much audio they were trained on.
  The checkpoint holds 1,555 pseudo-speakers; the 12 offered by name were selected by
  synthesising held-out text and scoring it with a phoneme recogniser. Hours turned out to be a
  poor predictor.
- **Batch generation is first-class.** Point it at a `.csv` of 50,000 lines and it parallelises,
  resumes, records a manifest, and doesn't abort the run because line 4,012 had a character it
  couldn't pronounce.

```bash
# Not on PyPI yet — install from source:
pip install "git+https://github.com/GhanaNLP/stable-twi-tts#egg=stable-twi-tts[hub,twi]"
apt install espeak-ng          # or: brew install espeak-ng   (needed for English)
```

The `twi` extra pulls ghana-g2p from source, because its wheel build is broken upstream. The
`hub` extra lets the model download itself:

```bash
stable-twi-tts --voice twi-6 --text "Akwaaba, wo ho te sɛn?" --out hello.wav
```

That fetches the published voice (~80 MB) on first run and caches it. Pass `--model <dir>` to use
a local voice directory instead.

## Listen

**[▶ Open the sample page](https://ghananlp.github.io/stable-twi-tts/)** — all clips with inline players. The links below open
GitHub's own audio preview.

### Every voice, same two sentences

So you can compare voices directly rather than across different content.

> **Twi** — *Akwaaba, wo ho te sɛn? Me da wo ase paa.*
> **Code-switched** — *Mepɛ sɛ mesua [computer science] wɔ [University of Ghana].*

| voice | Twi sample | code-switch sample | twi-only err | code-switch err | hours |
|---|---|---|---|---|---|
| `twi-1` | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_twi_twi-1.mp3) | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_cs_twi-1.mp3) | 34% | 60% | 6.9 |
| `twi-2` | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_twi_twi-2.mp3) | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_cs_twi-2.mp3) | 29% | 61% | 5.8 |
| `twi-3` | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_twi_twi-3.mp3) | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_cs_twi-3.mp3) | 29% | 61% | 11.2 |
| `twi-4` | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_twi_twi-4.mp3) | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_cs_twi-4.mp3) | 32% | 61% | 8.5 |
| `twi-5` | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_twi_twi-5.mp3) | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_cs_twi-5.mp3) | 28% | 61% | 5.9 |
| `twi-6` | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_twi_twi-6.mp3) | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_cs_twi-6.mp3) | 27% | 61% | 2.7 |
| `twi-7` | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_twi_twi-7.mp3) | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_cs_twi-7.mp3) | 27% | 61% | 3.3 |
| `twi-8` | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_twi_twi-8.mp3) | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_cs_twi-8.mp3) | 30% | 61% | 2.4 |
| `twi-9` | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_twi_twi-9.mp3) | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_cs_twi-9.mp3) | 28% | 66% | 3.5 |
| `twi-10` | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_twi_twi-10.mp3) | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_cs_twi-10.mp3) | 30% | 64% | 4.4 |
| `twi-11` | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_twi_twi-11.mp3) | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_cs_twi-11.mp3) | 30% | 63% | 9.5 |
| `twi-12` | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_twi_twi-12.mp3) | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/compare_cs_twi-12.mp3) | 30% | 64% | 3.4 |

Error figures are round-trip phoneme error: synthesise, re-recognise, compare against what was
asked for. Lower is better; the real-audio floor is 25.9% for Twi.

### Range of text

| | text | sample |
|---|---|---|
| **greeting** | Akwaaba! Yɛma wo akwaaba wɔ Ghana. | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/showcase_greeting.mp3) |
| **statement** | Ghana yɛ ɔman a ɛwɔ Afrika atɔeɛ fam. | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/showcase_statement.mp3) |
| **question** | Wo din de sɛn? Wofiri he na woreba? | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/showcase_question.mp3) |
| **long** | Anɔpa yi, ɔsoro abue na awia bɔ. Nnipa pii firi wɔn afie mu rekɔ adwuma, na mmɔfra nso rekɔ sukuu. | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/showcase_long.mp3) |
| **numbers** | Yɛn nsa kaa nnipa apem ne ahanum wɔ ɔmantam no mu. | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/showcase_numbers.mp3) |
| **news** | Ɔkyerɛkyerɛni no kaa sɛ [the examination] bɛba [next week]. | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/showcase_news.mp3) |
| **institution** | [Bank of Ghana] abɔ [interest rate] no so bio. | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/showcase_institution.mp3) |
| **english** | Good morning, and welcome to the news. | [▶ play](https://github.com/GhanaNLP/stable-twi-tts/raw/main/samples/showcase_english.mp3) |

**These are epoch-7 samples, not a finished voice.** Twi is solid; English is audibly weaker and
band-limited to 8 kHz because its training audio was 16 kHz where Twi's was 24 kHz.

## Web interface

A local GUI for people who would rather not use a terminal — and for turning documents into
speech without writing a script.

```bash
pip install "git+https://github.com/GhanaNLP/stable-twi-tts#egg=stable-twi-tts[web,twi]"
stable-twi-tts-web            # http://127.0.0.1:7860
```

Three ways in — **type text**, **upload a PDF**, or **give a URL** — and two ways out: a single
clip, or batch mode that splits into sentences and returns one joined wav or a zip of separate
files with a manifest.

Extraction and synthesis are deliberately separate steps: a PDF or web page becomes editable text
first, so you can fix a heading, drop a footer or bracket the English before anything is spoken.
Batch progress streams as it goes, because a 40-page PDF otherwise looks like a hang. The voice
picker shows each voice's measured error, since that is the part worth choosing on.

It binds to **127.0.0.1** by default. `--host 0.0.0.0` exposes it, and the URL-fetch endpoint then
makes requests from your machine on a caller's behalf — private, loopback and link-local addresses
are refused, but do not put this on an untrusted network.

## Speak something

```bash
stable-twi-tts --text "Akwaaba, wo ho te sɛn?" --out hello.wav
stable-twi-tts --language eng --text "Good morning, Accra." --out en.wav
```

```python
from stable_twi_tts import StableTwiTTS

tts = StableTwiTTS.from_pretrained()          # downloads and caches the published voice
tts.synthesize("Akwaaba, wo ho te sɛn?", voice="twi-6").save("hello.wav")

# or point at a local voice directory
tts = StableTwiTTS("voices/my_voice")
```

## Choosing a voice

```bash
stable-twi-tts --list-voices
```

```
voice    lang   hours  code-switch  twi-only  source
twi-1    twi     6.88        59.8%     33.5%  spk_0016
twi-2    twi     5.80        60.7%     29.1%  spk_0080
twi-3    twi    11.24        61.0%     29.1%  spk_0006
...
twi-6    twi     2.72        61.2%     26.8%  spk_0002
twi-9    twi     3.51        65.5%     28.1%  spk_0165
```

**Voices are ranked by measured intelligibility, not by training hours** — and the two disagree
sharply. Each voice synthesised the same held-out text, which was then re-recognised and scored
for phoneme error: Twi with the Ghana phoneme ASR, the English spans with KoelLabs. 30 of the
207 Twi pseudo-speakers were measured.

Two rankings, because they do not substitute for each other:

| use | tier | pick |
|---|---|---|
| text mixing English into Twi | `tiers.codeswitch` | `twi-1` (59.8%) |
| pure Twi | `tiers.twi_only` | `twi-6` (26.8%, floor is 25.9%) |

`twi-1` is the **best code-switch voice but 21st of 30 on pure Twi**; `twi-9` is 3rd on Twi and
among the worst on code-switch. Had we ranked by hours — as the first version did — the best
mixed-text voice would have been buried and two of the three best Twi voices excluded entirely
(they have under 3.3 h each).

`voices.json` records every measurement, so the ranking is inspectable rather than asserted.
Unlisted speakers are reachable by raw index (`--voice 42`), unmeasured.

There are **no separate English voices.** These are Twi voices judged on how well they also
handle English, which is what reading real Ghanaian text requires.

## Batch generation

Any of `.txt` (one utterance per line), `.csv`/`.tsv` (needs a `text` column), or `.jsonl`.

```bash
stable-twi-tts --input corpus.csv --out synth/ --workers 8
```

```
50000 utterances -> synth/  (8 workers)
  [25/50000] 14.2/s  eta 58.7 min
  ...
49987 written, 0 skipped, 13 failed in 57.4 min
41.20 h of audio  (2586x realtime)
manifest: synth/manifest.jsonl
```

Per-row columns override the defaults, so one file can mix voices and languages:

```csv
id,text,voice,language
greet_01,"Akwaaba, wo ho te sɛn?",twi-1,twi
news_01,"Good morning, welcome to the news.",eng-1,eng
mix_01,"Mepɛ sɛ mesua [computer science].",twi-2,mixed
```

Every run writes `manifest.jsonl`, one record per utterance:

```json
{"id": "greet_01", "status": "ok", "path": "synth/greet_01.wav", "duration": 1.83, "voice": "twi-1", "n_phonemes": 17}
{"id": "bad_09", "status": "error", "error": "PhonemeError: the model has no symbol for: ['ʈ']"}
```

Re-running skips what already exists, so an interrupted 50k job resumes where it stopped
(`--overwrite` to force). Failures are per-item — a corpus of 50,000 lines does not lose 49,999
outputs because one line was unpronounceable.

## Text really does become phonemes

The model reads phoneme ids, never letters, and it only knows the phonemes its training targets
contained. So this package reproduces the training front-end exactly:

| language | phonemiser |
|---|---|
| Twi | [ghana-g2p](https://github.com/GhanaNLP/ghana-g2p) — the same library that produced the Twi targets |
| English | espeak-ng `en-us` — the same phonemiser that produced the English targets |

**Do not substitute a different English G2P.** This is the one change most likely to quietly
ruin output. An earlier version of this pipeline folded English into the Ghanaian inventory
(`θ`→`t`, `æ`→`a`). It produced valid-looking IPA, disagreed with the training targets on **51%
of units**, and the model scored **68.6%** phoneme error where Twi scored 25.6%. Nothing errored;
it just sounded wrong. Same function on both sides is the difference between working and not.

Note that the *accent* lives in the audio, not the symbols. English input is canonical
(`θ æ ɹ eɪ`), and the Ghanaian accent comes from the voice.

## Two deployment paths

| path | phoneme error | needs |
|---|---|---|
| **Python + onnxruntime** | **30.2%** | Python, ghana-g2p, espeak-ng for English — unlimited vocabulary |
| **Native port, no espeak** | matches Python | ~30 lines + a 0.88 MB English lexicon — no Python, no native linking |
| Native port + `libespeak-ng` | matches Python | ~30 lines + 1.56 MB — adds unlimited English vocabulary |

The native path exists because the Python dependency is only the *front-end*, and Twi's
grapheme-to-phoneme is a **42-entry longest-match table** — verified identical to ghana-g2p across
20,000 words. So a Twi-only app needs no Python, no espeak and no lexicon. English cannot be ported that way (7,132
context rules plus 5,794 exceptions), but it does not need to be: `mobile/english_lexicon.json.gz`
holds 124,926 precomputed pronunciations in 0.88 MB and covers **98.2% of tokens** in real text.
Link `libespeak-ng` (1.56 MB) only if you need unlimited English vocabulary.

`mobile/` has Kotlin and Swift references plus **13 test vectors**, so a port is verified rather
than hoped-for. Those vectors are worth taking seriously: writing them caught two bugs in the
specification that a 20,000-word equivalence test had missed, and both would have produced
fluent-sounding wrong audio rather than an error.

<details>
<summary>A third path exists via sherpa-onnx, but neither of the above needs it</summary>

sherpa-onnx can load `model.onnx` directly, but it does its own text-to-phoneme step and
**espeak-ng has no Twi**, so it needs the generated `lexicon_ascii.txt` plus two string
replacements (`ɔ`→`q`, `ɛ`→`x` — Akan has no q or x, so they cannot collide with a real word).
It measured ~36–38% phoneme error and covers 78k words, silently dropping anything outside that.
Superseded by the native port, which is both better and simpler.

</details>

## Building a voice directory

```bash
python tools/export_voice.py \
    --checkpoint runs/piper/checkpoints/best.ckpt \
    --train-config runs/piper/config.json \
    --manifest data/manifest.tsv \
    --out voices/my_voice \
    --top-n 10 --min-hours 1.0 --lexicon
```

Produces `model.onnx`, `config.json`, `voices.json`, `tokens.txt` and optionally `lexicon.txt`.
Voices are ranked by **hours, not clip count** — Twi clips average 3.9 s and English 13.6 s, so
ranking by clips would systematically flatter Twi speakers.

## Honest status

- **Not on PyPI.** Install from git; the package name is not registered yet.
- **The model is epoch 7 of a run that was stopped early**, not a converged model. `val_mos` was
  still climbing (2.56 → 3.02) when training stopped. A longer run should be better.
- **The Kotlin and Swift ports have not been compiled** — the algorithm is verified against test
  vectors, but no one has built them on a device yet.
- **English is markedly weaker than Twi** and this is a data problem, not a tuning one: English had
  a third of the utterances (43k vs 151k) at 3.5× the length, and VITS learns alignment per
  utterance.

## Known limits

- **English voices are band-limited to 8 kHz.** The English training audio was 16 kHz; the Twi
  was 24 kHz. Upsampling cannot invent the missing highs, so English voices sound duller than
  Twi ones. This is a property of the training data, not a bug.
- **Code-switching is extrapolation.** No training utterance mixed languages within a sentence.
  It works, but it is not a trained capability, and the language token is per-utterance so a
  switched sentence has to pick one frame language.
- **Speakers are derived, not real identities.** Pseudo-speakers from clustering. One real
  person may appear as two voices, and a voice is not a consented identity.
- **Long input is split on sentence boundaries** and concatenated with a short pause. Training
  clips were 4–14 s and VITS attention degrades well beyond that.

## License

Code MIT. The model and voices carry the licences of their training data — the Twi half is
`cc-by-nc-4.0`, so **the voices are non-commercial**.

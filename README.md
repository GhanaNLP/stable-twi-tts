# twi-ipa-tts

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
pip install twi-ipa-tts
apt install espeak-ng                      # needed for English
pip install 'twi-ipa-tts[twi]'               # needed for Twi
```

## Speak something

```bash
twi-ipa-tts --model voices/twi-ipa --text "Akwaaba, wo ho te sɛn?" --out hello.wav
twi-ipa-tts --model voices/twi-ipa --language eng --text "Good morning, Accra." --out en.wav
```

```python
from twi_ipa_tts import TwiIpaTTS

tts = TwiIpaTTS("voices/twi-ipa")
tts.synthesize("Akwaaba, wo ho te sɛn?", voice="twi-1").save("hello.wav")
```

## Choosing a voice

```bash
twi-ipa-tts --model voices/twi-ipa --list-voices
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
twi-ipa-tts --model voices/twi-ipa --input corpus.csv --out synth/ --workers 8
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

## Running without Python (sherpa-onnx, mobile, C++)

`model.onnx` is a standard Piper VITS export, so [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
can run it on Android, iOS, C++, C#, Go and the rest. There is one real constraint to understand.

sherpa-onnx does its own text-to-phoneme step, from either espeak-ng data or a lexicon file.
**espeak-ng has no Asante Twi**, so the espeak route cannot produce correct Twi. The lexicon
route can, which is why `export_voice.py --lexicon` ships `lexicon.txt`:

```python
import sherpa_onnx

tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model="voices/twi-ipa/model.onnx",
            tokens="voices/twi-ipa/tokens.txt",
            lexicon="voices/twi-ipa/lexicon.txt",
        ))))
audio = tts.generate("Akwaaba", sid=0)
```

A lexicon only covers the words it was generated from. **Words outside it will be silently
skipped**, where the Python front-end calls ghana-g2p and pronounces anything, including words
it has never seen. So: use the Python front-end wherever Python is available, and the lexicon
route when you need a native runtime and can accept fixed vocabulary.

## Building a voice directory

```bash
python tools/export_voice.py \
    --checkpoint runs/piper/checkpoints/best.ckpt \
    --train-config runs/piper/config.json \
    --manifest data/manifest.tsv \
    --out voices/twi-ipa \
    --top-n 10 --min-hours 1.0 --lexicon
```

Produces `model.onnx`, `config.json`, `voices.json`, `tokens.txt` and optionally `lexicon.txt`.
Voices are ranked by **hours, not clip count** — Twi clips average 3.9 s and English 13.6 s, so
ranking by clips would systematically flatter Twi speakers.

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

# Running on mobile, without Python

The model is ONNX, so it runs wherever onnxruntime does — including Android and iOS. The only
Python in the pipeline is the **front-end** that turns text into phoneme ids, and that is portable:

| step | on device | cost |
|---|---|---|
| Twi text → phonemes | port the 42-rule table below | **~30 lines of code** |
| English text → phonemes | link `libespeak-ng`, English data only | **1.56 MB** |
| phonemes → ids | table lookup + wrapping | ~20 lines |
| ids → audio | onnxruntime Android / iOS | — |

Twi is a 42-rule port and needs nothing else. English links `libespeak-ng`, which
`tools/bundle_espeak_english.sh` reduces from the full 31 MB install to the 1.56 MB English
actually needs. A lexicon alternative exists for targets that cannot link native code — see below.

For a **Twi-only** app you need no espeak and no lexicon at all: the 42-rule port gives unlimited
vocabulary. That is strictly better than the sherpa-onnx lexicon route, which caps you at the
words the lexicon was generated from.

## English: link espeak

English cannot be *ported* the way Twi can, so link the library:

```bash
tools/bundle_espeak_english.sh espeak-english     # builds and strips to 1.56 MB
```

```
libespeak-ng.so.1.53.0   715 KB      phondata           600 KB
en_dict                  182 KB      phontab             64 KB
phonindex                 49 KB      phondata-manifest   23 KB
intonations                3 KB      lang/gmw/en        140 B
                                     TOTAL             1.56 MB
```

Verified working from the bundle alone. Point `ESPEAK_DATA_PATH` at `espeak-ng-data/` and call
espeak with IPA output.

| | size | vocabulary | needs |
|---|---|---|---|
| **`libespeak-ng` + en data** | **1.56 MB** | **unlimited** | native linking |
| `english_lexicon.json.gz` | 0.88 MB | 98.2% of tokens | nothing |

**Link the library.** The 0.68 MB saved by the lexicon is not worth silently dropping words —
and what it drops is exactly what matters: proper nouns, inflections, anything coined recently.
The lexicon is the fallback for targets where native linking is impossible (a pure-JS or
sandboxed runtime), not the default.

**And coverage is not the only gap — measured, not assumed.** Comparing the lexicon against
`espeak-ng -q --ipa -v en-us` directly:

| | lexicon vs espeak |
|---|---|
| single words (3,000 sampled) | **5.7% differ** |
| whole sentences | **every one differs** |

Two separate causes. Single-word differences are espeak's own context rules firing on a word in
isolation versus in a list (`mentors` → `mˈɛntɔːɹz` vs `mˈɛntoːɹz`, `wealthiest` →
`wˈɛlθiʲɪst` vs `wˈɛlθiɪst`). Sentences differ for a deeper reason: espeak reduces function words
in context, so *to* is `tə` not `tˈuː`, *the* is `ðə` not `ðˈə`, and *and* loses its stress
entirely. A per-word lexicon cannot know it is in a sentence, so it stresses every word.

The model was trained on espeak's sentence-level output. A front-end that stresses every function
word is a front-end that disagrees with training on most utterances — the same class of mismatch
that produced 68.6% phoneme error here once already. So the lexicon is genuinely a last resort:
correct enough for isolated words, measurably wrong for connected speech.

### The lexicon fallback

124,926 words, already tokenised into the model's units — split on spaces, do not re-tokenise:

```json
{"computer": "k ə m p j ˈ uː ɾ ɚ", "ghana": "ɡ ˈ ɑ ː n ə", ...}
```

Coverage was measured on real Ghanaian English broadcast text, not asserted: **98.2% of tokens,
85% of types**. The gap between those two numbers is the point — the misses are rare words, and
what remains missing is mostly ASR noise (`dignityity`), Ghanaian proper nouns (`takoradi`) and
possessives (`bia's`). Add your own domain vocabulary with
`tools/build_english_lexicon.py --extra-words`.

Words absent from the lexicon **cannot be pronounced** and must be skipped or spelled out — which
is why it is the fallback rather than the default.

**Licence note:** espeak-ng is GPL-3.0, so linking it into an application carries obligations that
shipping a generated lexicon does not. That is a genuine reason to prefer the lexicon in some
projects, independent of size.

### espeak is still the source of truth

The lexicon was generated *by* espeak-ng, once. That is not incidental: the model was trained on
`espeak-ng -v en-us --ipa` output, so pronunciations from any other source — CMUdict's ARPAbet, a
different synthesiser, a hand-written table — would be *different phonemes* and reintroduce a
train/inference mismatch. That mismatch is not theoretical: at 51% disagreement it produced 68.6%
phoneme error where Twi scored 25.6%, with nothing raising an error.

So espeak is a build-time dependency for whoever regenerates the lexicon, and no dependency at all
for whoever ships the app.

## Why Twi ports and English doesn't

```
Twi:      42 rules,     0 exceptions   — context-free, orthography ≈ phonemic
English:  7,132 rules,  5,794 exceptions — deeply context-sensitive
```

Twi's orthography is near-phonemic, so longest-match over a table *is* the algorithm — verified
identical to ghana-g2p across 20,000 words. English needs `ough` to differ in *though*, *through*,
*cough* and *bough*, which is what espeak's 7,132 context rules and exception dictionary encode.
Reimplementing that means reimplementing espeak.

**Use the same espeak-ng the model was trained with.** We trained English on
`espeak-ng -q --ipa -v en-us`. Classic eSpeak, a fork, or any reimplementation emits different
phonemes — which does not error, it just degrades output. That mismatch already cost this project a
day: a front-end disagreeing with training on 51% of units produced 68.6% phoneme error where Twi
scored 25.6%.

## Files

| file | what |
|---|---|
| `twi_rules.json` | the 42-entry Twi table, as data — ports carry no hand-copied version |
| `test_vectors.json` | 13 cases: text → expected units → expected ids |
| `verify_vectors.py` | independent reimplementation, checked against the vectors |
| `kotlin/TwiFrontend.kt` | Android reference |
| `swift/TwiFrontend.swift` | iOS reference |
| `gen_test_vectors.py` | regenerates the vectors from the Python front-end |

## Verify before you trust it

```bash
python verify_vectors.py --dir . --config ../voices/stable-twi-tts/config.json
# 13 passed, 0 failed, 0 skipped
```

`verify_vectors.py` imports nothing from `stable_twi_tts` — it reads only `twi_rules.json`, the
model's `config.json` and espeak, exactly as a port does. Use it as the template for your port's
own test, and make your port reproduce `units` and `ids` for all 13 cases.

**This matters more than it sounds.** Every step here fails *silently*: a dropped pad halves the
sequence, a split multi-character symbol becomes an unknown unit, an emitted space feeds the model
something it never trained on. None of those throw — they produce fluent-sounding wrong audio.
Writing these vectors caught two bugs in my own specification that a 20,000-word test had missed.

## The five conventions a port must match

1. **Longest match first.** `kʰ`, `t͡ʃ`, `aɪ`, `eɪ`, `iː` are single units, and `ky` must be tried
   before `k`. Splitting by codepoint shatters them into units the model has never seen.
2. **Skip whitespace; keep punctuation.** The model *has* a space symbol (id 3) but the training
   targets never used it, so emitting it is wrong. Punctuation (`.` `,` `?`) *was* trained on and
   must pass through — it carries prosody.
3. **Keep stress and length marks.** espeak emits `ˈ ˌ ː` and they are in the inventory. Stripping
   them changes the input distribution.
4. **Wrap as `BOS PAD (unit PAD)* EOS`.** The pad between *every* unit is not decoration; omitting
   it halves the sequence the duration predictor sees.
5. **Language token after the leading PAD**, when the model has one. Mixed text uses the **Twi**
   token, because Twi is the matrix language in Ghanaian code-switching — tagging Twi phonemes as
   English asks for a combination never trained.

Twi input is lowercased; English is passed to espeak unchanged.

## Android sketch

```kotlin
val fe = TwiFrontend(assets.open("twi_rules.json").reader().readText(),
                     JSONObject(configJson).getJSONObject("phoneme_id_map").toString())

val ids = fe.twiToIds("Akwaaba, wo ho te sɛn?")          // no espeak needed
// or, with espeak linked for English:
val ids = fe.mixedToIds("Mepɛ sɛ mesua [computer science].") { text -> espeakIpa(text) }

val env = OrtEnvironment.getEnvironment()
val session = env.createSession(modelBytes)
val input = OnnxTensor.createTensor(env, arrayOf(ids.map { it.toLong() }.toLongArray()))
val lengths = OnnxTensor.createTensor(env, longArrayOf(ids.size.toLong()))
val scales = OnnxTensor.createTensor(env, floatArrayOf(0.667f, 1.0f, 0.8f))  // noise, length, noise_w
val sid = OnnxTensor.createTensor(env, longArrayOf(speakerId.toLong()))
val audio = session.run(mapOf("input" to input, "input_lengths" to lengths,
                              "scales" to scales, "sid" to sid))
```

Speaker ids come from `voices.json` — use `tiers.codeswitch` for mixed text and `tiers.twi_only`
for pure Twi, since the two rankings disagree sharply.

## Building English-only espeak

The 1.56 MB figure is the shared library plus only the data English needs:

```
libespeak-ng.so     715 KB
phondata            600 KB
en_dict             182 KB
phontab              64 KB
phonindex            49 KB
phondata-manifest    23 KB
intonations           3 KB
lang/gmw/en          140 B
```

espeak-ng builds with CMake and has official Android support — this is what Piper's own Android
app does. Call it with the IPA output flag so it matches `--ipa` from the CLI; confirm that once
against `test_vectors.json` rather than assuming, since the API and CLI are different call paths.

## If porting proves awkward

The sherpa-onnx route still works and needs no port: `lexicon_ascii.txt` plus two string
replacements (`ɔ`→`q`, `ɛ`→`x`, because Akan has no q or x so they cannot collide). It measured
~36–38% phoneme error against the Python path's 30%, and covers 78k words with unknown words
dropped. Slower to improve, but available today.

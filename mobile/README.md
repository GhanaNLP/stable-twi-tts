# Running on mobile, without Python

The model is ONNX, so it runs wherever onnxruntime does — including Android and iOS. The only
Python in the pipeline is the **front-end** that turns text into phoneme ids, and that is portable:

| step | on device | cost |
|---|---|---|
| Twi text → phonemes | port the 42-rule table below | **~30 lines of code** |
| English text → phonemes | link `libespeak-ng`, English data only | **1.56 MB** |
| phonemes → ids | table lookup + wrapping | ~20 lines |
| ids → audio | onnxruntime Android / iOS | — |

For a **Twi-only** app you need no espeak and no lexicon at all: the 42-rule port gives unlimited
vocabulary. That is strictly better than the sherpa-onnx lexicon route, which caps you at the
words the lexicon was generated from.

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

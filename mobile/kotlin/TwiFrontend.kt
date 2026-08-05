package org.ghananlp.stabletwitts

import org.json.JSONObject

/**
 * Text -> phoneme ids for the Twi TTS model, with no Python.
 *
 * Twi needs no external phonemiser: its orthography is near-phonemic, so grapheme-to-phoneme is a
 * 42-entry longest-match table. This implementation was verified identical to ghana-g2p (the
 * Python reference) across 20,000 words.
 *
 * English is not portable this way — espeak-ng's English is 7,132 context-sensitive rules plus
 * 5,794 exceptions — so English text must go through libespeak-ng, which is ~1.6 MB with
 * English-only data. Use the *same* espeak-ng the model was trained with; classic eSpeak or any
 * reimplementation emits different phonemes, which degrades output without erroring.
 *
 * Verify any change against mobile/test_vectors.json. Every step here has a failure mode that
 * produces fluent-sounding wrong audio rather than an exception.
 */
class TwiFrontend(
    /** Contents of twi_rules.json. */
    twiRulesJson: String,
    /** The model's phoneme_id_map, from config.json. */
    phonemeIdMapJson: String,
) {
    private val graphemes: List<Pair<String, String>>
    private val idMap: Map<String, List<Int>>
    private val symbolsByLength: List<String>

    companion object {
        private const val PAD = "_"
        private const val BOS = "^"
        private const val EOS = "$"
    }

    init {
        val rules = JSONObject(twiRulesJson).getJSONObject("graphemes")
        // Longest first: 'ky' must be tried before 'k', or it is never reached.
        graphemes = rules.keys().asSequence()
            .map { it to rules.getString(it) }
            .sortedByDescending { it.first.length }
            .toList()

        val ids = JSONObject(phonemeIdMapJson)
        idMap = ids.keys().asSequence().associateWith { key ->
            val v = ids.get(key)
            if (v is Int) listOf(v)
            else (v as org.json.JSONArray).let { arr -> (0 until arr.length()).map { arr.getInt(it) } }
        }
        // Blank symbols are excluded: the model has a space symbol, but the training targets
        // never used it, and it cannot survive a space-separated phoneme string.
        symbolsByLength = idMap.keys.filter { it.isNotBlank() }.sortedByDescending { it.length }
    }

    /** Twi text -> phoneme units. Lowercases first, as the reference does. */
    fun twiPhonemes(text: String): List<String> {
        val out = ArrayList<String>()
        val w = text.lowercase()
        var i = 0
        outer@ while (i < w.length) {
            for ((graph, ipa) in graphemes) {
                if (w.startsWith(graph, i)) {
                    out.add(ipa)
                    i += graph.length
                    continue@outer
                }
            }
            // No grapheme rule. Pass the character through if the model has a symbol for it:
            // this is what carries punctuation, which the model was trained with as units.
            // Punctuation yes, whitespace no: the model has a space symbol but was never
            // trained with it, so emitting it feeds sequences unlike anything seen.
            if (!w[i].isWhitespace() && idMap.containsKey(w[i].toString())) out.add(w[i].toString())
            i++
        }
        return out
    }

    /**
     * Split an IPA string into the model's units, greedy longest-first.
     *
     * Whitespace is skipped, never emitted. espeak separates words with spaces and the model does
     * have a space symbol, but the English targets were built without it — emitting it feeds
     * sequences unlike anything trained.
     */
    fun tokenize(ipa: String): List<String> {
        val out = ArrayList<String>()
        var i = 0
        outer@ while (i < ipa.length) {
            if (ipa[i].isWhitespace()) { i++; continue }
            for (sym in symbolsByLength) {
                if (ipa.startsWith(sym, i)) {
                    out.add(sym)
                    i += sym.length
                    continue@outer
                }
            }
            i++   // a codepoint the model has no symbol for; dropping beats guessing
        }
        return out
    }

    /**
     * Units -> ids: BOS, PAD, then each unit followed by PAD, then EOS.
     *
     * The pad between every unit is not decoration — the model was trained with it, and omitting
     * it halves the sequence the duration predictor sees.
     */
    fun toIds(units: List<String>, language: String): IntArray {
        require(units.isNotEmpty()) { "no phonemes to speak" }
        val missing = units.filter { it !in idMap }
        require(missing.isEmpty()) { "model has no symbol for: $missing" }

        val ids = ArrayList<Int>(units.size * 2 + 4)
        ids.addAll(idMap.getValue(BOS))
        ids.addAll(idMap.getValue(PAD))
        // Per-utterance language token, present only in bilingual models. Mixed text uses the
        // Twi token, since Twi is the matrix language in Ghanaian code-switching.
        val tok = "«${if (language == "mixed") "twi" else language}»"
        idMap[tok]?.let { ids.addAll(it); ids.addAll(idMap.getValue(PAD)) }
        for (u in units) {
            ids.addAll(idMap.getValue(u))
            ids.addAll(idMap.getValue(PAD))
        }
        ids.addAll(idMap.getValue(EOS))
        return ids.toIntArray()
    }

    /** Convenience: Twi text straight to ids. */
    fun twiToIds(text: String): IntArray = toIds(twiPhonemes(text), "twi")

    /**
     * English or code-switched text to ids.
     *
     * [espeakIpa] must call libespeak-ng with IPA output for the en-us voice, i.e. the equivalent
     * of `espeak-ng -q --ipa -v en-us`. Supplying a different phonemiser is the single most
     * effective way to ruin output while everything appears to work.
     */
    fun englishToIds(text: String, espeakIpa: (String) -> String): IntArray =
        toIds(tokenize(espeakIpa(text)), "eng")

    /**
     * Code-switched text: [bracketed] spans are English inside a Twi frame.
     *
     * Each span is phonemised by its own language, because the two halves of the training data use
     * different sub-inventories — Twi is aspirated (kʰ, pʰ, tʰ, ɾ) and English is not (k, p, t, r).
     */
    fun mixedToIds(text: String, espeakIpa: (String) -> String): IntArray {
        val units = ArrayList<String>()
        val re = Regex("\\[([^\\]]*)\\]")
        var pos = 0
        for (m in re.findAll(text)) {
            val head = text.substring(pos, m.range.first).trim()
            if (head.isNotEmpty()) units.addAll(twiPhonemes(head))
            val span = m.groupValues[1].trim()
            if (span.isNotEmpty()) units.addAll(tokenize(espeakIpa(span)))
            pos = m.range.last + 1
        }
        val tail = text.substring(pos).trim()
        if (tail.isNotEmpty()) units.addAll(twiPhonemes(tail))
        return toIds(units, "mixed")
    }
}

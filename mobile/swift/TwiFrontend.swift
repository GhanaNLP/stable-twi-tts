import Foundation

/// Text -> phoneme ids for the Twi TTS model, with no Python.
///
/// Twi needs no external phonemiser: its orthography is near-phonemic, so grapheme-to-phoneme is
/// a 42-entry longest-match table. This implementation was verified identical to ghana-g2p (the
/// Python reference) across 20,000 words.
///
/// English is not portable this way — espeak-ng's English is 7,132 context-sensitive rules plus
/// 5,794 exceptions — so English text must go through libespeak-ng, which is ~1.6 MB with
/// English-only data. Use the *same* espeak-ng the model was trained with; classic eSpeak or any
/// reimplementation emits different phonemes, which degrades output without erroring.
///
/// Verify any change against mobile/test_vectors.json. Every step here has a failure mode that
/// produces fluent-sounding wrong audio rather than throwing.
public struct TwiFrontend {

    private let graphemes: [(graph: String, ipa: String)]   // longest first
    private let idMap: [String: [Int]]
    private let symbolsByLength: [String]

    private static let pad = "_"
    private static let bos = "^"
    private static let eos = "$"

    public enum FrontendError: Error, CustomStringConvertible {
        case emptyInput
        case unknownSymbols([String])

        public var description: String {
            switch self {
            case .emptyInput: return "no phonemes to speak"
            case .unknownSymbols(let s): return "model has no symbol for: \(s)"
            }
        }
    }

    /// - Parameters:
    ///   - twiRules: `graphemes` object from twi_rules.json
    ///   - phonemeIdMap: `phoneme_id_map` from the model's config.json
    public init(twiRules: [String: String], phonemeIdMap: [String: [Int]]) {
        // Longest first: "ky" must be tried before "k", or it is never reached.
        graphemes = twiRules
            .map { (graph: $0.key, ipa: $0.value) }
            .sorted { $0.graph.count > $1.graph.count }
        idMap = phonemeIdMap
        // Blank symbols are excluded: the model has a space symbol, but the training targets
        // never used it, and it cannot survive a space-separated phoneme string.
        symbolsByLength = phonemeIdMap.keys
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .sorted { $0.count > $1.count }
    }

    /// Twi text -> phoneme units. Lowercases first, as the reference does.
    public func twiPhonemes(_ text: String) -> [String] {
        var out: [String] = []
        let w = Array(text.lowercased())
        var i = 0
        outer: while i < w.count {
            for rule in graphemes {
                let g = Array(rule.graph)
                if i + g.count <= w.count && Array(w[i..<(i + g.count)]) == g {
                    out.append(rule.ipa)
                    i += g.count
                    continue outer
                }
            }
            // No grapheme rule. Pass the character through if the model has a symbol for it:
            // this is what carries punctuation, which the model was trained with as units.
            // Punctuation yes, whitespace no: the model has a space symbol but was never
            // trained with it, so emitting it feeds sequences unlike anything seen.
            if !w[i].isWhitespace, idMap[String(w[i])] != nil { out.append(String(w[i])) }
            i += 1
        }
        return out
    }

    /// Split an IPA string into the model's units, greedy longest-first.
    ///
    /// Whitespace is skipped, never emitted. espeak separates words with spaces and the model does
    /// have a space symbol, but the English targets were built without it — emitting it feeds
    /// sequences unlike anything trained.
    public func tokenize(_ ipa: String) -> [String] {
        var out: [String] = []
        let chars = Array(ipa)
        var i = 0
        outer: while i < chars.count {
            if chars[i].isWhitespace { i += 1; continue }
            for sym in symbolsByLength {
                let s = Array(sym)
                if i + s.count <= chars.count && Array(chars[i..<(i + s.count)]) == s {
                    out.append(sym)
                    i += s.count
                    continue outer
                }
            }
            i += 1   // a codepoint the model has no symbol for; dropping beats guessing
        }
        return out
    }

    /// Units -> ids: BOS, PAD, then each unit followed by PAD, then EOS.
    ///
    /// The pad between every unit is not decoration — the model was trained with it, and omitting
    /// it halves the sequence the duration predictor sees.
    public func toIds(_ units: [String], language: String) throws -> [Int] {
        guard !units.isEmpty else { throw FrontendError.emptyInput }
        let missing = units.filter { idMap[$0] == nil }
        guard missing.isEmpty else { throw FrontendError.unknownSymbols(Array(Set(missing)).sorted()) }

        var ids: [Int] = []
        ids.reserveCapacity(units.count * 2 + 4)
        ids += idMap[Self.bos]!
        ids += idMap[Self.pad]!
        // Per-utterance language token, present only in bilingual models. Mixed text uses the Twi
        // token, since Twi is the matrix language in Ghanaian code-switching.
        let tok = "«\(language == "mixed" ? "twi" : language)»"
        if let t = idMap[tok] {
            ids += t
            ids += idMap[Self.pad]!
        }
        for u in units {
            ids += idMap[u]!
            ids += idMap[Self.pad]!
        }
        ids += idMap[Self.eos]!
        return ids
    }

    /// Convenience: Twi text straight to ids.
    public func twiToIds(_ text: String) throws -> [Int] {
        try toIds(twiPhonemes(text), language: "twi")
    }

    /// English text to ids.
    ///
    /// `espeakIpa` must call libespeak-ng with IPA output for the en-us voice, i.e. the equivalent
    /// of `espeak-ng -q --ipa -v en-us`. Supplying a different phonemiser is the single most
    /// effective way to ruin output while everything appears to work.
    public func englishToIds(_ text: String,
                            espeakIpa: (String) -> String) throws -> [Int] {
        try toIds(tokenize(espeakIpa(text)), language: "eng")
    }

    /// Code-switched text: `[bracketed]` spans are English inside a Twi frame.
    ///
    /// Each span is phonemised by its own language, because the two halves of the training data use
    /// different sub-inventories — Twi is aspirated (kʰ, pʰ, tʰ, ɾ) and English is not (k, p, t, r).
    public func mixedToIds(_ text: String,
                           espeakIpa: (String) -> String) throws -> [Int] {
        var units: [String] = []
        let re = try! NSRegularExpression(pattern: "\\[([^\\]]*)\\]")
        let ns = text as NSString
        var pos = 0
        for m in re.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let head = ns.substring(with: NSRange(location: pos, length: m.range.location - pos))
                .trimmingCharacters(in: .whitespaces)
            if !head.isEmpty { units += twiPhonemes(head) }
            let span = ns.substring(with: m.range(at: 1))
                .trimmingCharacters(in: .whitespaces)
            if !span.isEmpty { units += tokenize(espeakIpa(span)) }
            pos = m.range.location + m.range.length
        }
        let tail = ns.substring(from: pos).trimmingCharacters(in: .whitespaces)
        if !tail.isEmpty { units += twiPhonemes(tail) }
        return try toIds(units, language: "mixed")
    }
}

import Foundation

public struct RampartToken: Equatable, Sendable {
    public let text: String
    public let id: Int32
    public let range: Range<String.Index>?
    public let isSpecialToken: Bool

    public init(text: String, id: Int32, range: Range<String.Index>?, isSpecialToken: Bool) {
        self.text = text
        self.id = id
        self.range = range
        self.isSpecialToken = isSpecialToken
    }
}

public struct RampartEncodedInput: Sendable {
    public let tokens: [RampartToken]
    public let inputIDs: [Int32]
    public let attentionMask: [Int32]
    public let tokenTypeIDs: [Int32]
    public let activeTokenCount: Int
    public let truncated: Bool
}

public enum RampartTokenizerError: Error, LocalizedError {
    case missingToken(String)
    case invalidVocabLine(String)

    public var errorDescription: String? {
        switch self {
        case .missingToken(let token):
            "Vocabulary is missing required token \(token)."
        case .invalidVocabLine(let line):
            "Invalid vocabulary line: \(line)"
        }
    }
}

public final class BertWordPieceTokenizer: Sendable {
    public let vocabulary: [String: Int32]
    public let padToken: String
    public let unknownToken: String
    public let clsToken: String
    public let sepToken: String
    public let maskToken: String
    public let maxInputCharactersPerWord: Int

    public init(
        vocabulary: [String: Int32],
        padToken: String = "[PAD]",
        unknownToken: String = "[UNK]",
        clsToken: String = "[CLS]",
        sepToken: String = "[SEP]",
        maskToken: String = "[MASK]",
        maxInputCharactersPerWord: Int = 100
    ) throws {
        self.vocabulary = vocabulary
        self.padToken = padToken
        self.unknownToken = unknownToken
        self.clsToken = clsToken
        self.sepToken = sepToken
        self.maskToken = maskToken
        self.maxInputCharactersPerWord = maxInputCharactersPerWord

        for token in [padToken, unknownToken, clsToken, sepToken, maskToken] {
            guard vocabulary[token] != nil else {
                throw RampartTokenizerError.missingToken(token)
            }
        }
    }

    public convenience init(vocabURL: URL) throws {
        let contents = try String(contentsOf: vocabURL, encoding: .utf8)
        var vocabulary: [String: Int32] = [:]
        for (index, rawLine) in contents.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            let token = rawLine.trimmingCharacters(in: .newlines)
            guard !token.isEmpty else { continue }
            guard vocabulary[token] == nil else {
                throw RampartTokenizerError.invalidVocabLine(token)
            }
            vocabulary[token] = Int32(index)
        }
        try self.init(vocabulary: vocabulary)
    }

    public var padTokenID: Int32 {
        vocabulary[padToken]!
    }

    public func encode(_ text: String, sequenceLength: Int = 512) -> RampartEncodedInput {
        let wordPieces = contentTokens(text)

        var tokens: [RampartToken] = [
            RampartToken(text: clsToken, id: vocabulary[clsToken]!, range: nil, isSpecialToken: true)
        ]
        tokens.append(contentsOf: wordPieces)
        tokens.append(RampartToken(text: sepToken, id: vocabulary[sepToken]!, range: nil, isSpecialToken: true))

        let truncated = tokens.count > sequenceLength
        if truncated {
            tokens = Array(tokens.prefix(sequenceLength))
            if let sepID = vocabulary[sepToken] {
                tokens[sequenceLength - 1] = RampartToken(
                    text: sepToken,
                    id: sepID,
                    range: nil,
                    isSpecialToken: true
                )
            }
        }

        let activeTokenCount = tokens.count
        var inputIDs = tokens.map(\.id)
        var attentionMask = Array(repeating: Int32(1), count: activeTokenCount)
        var tokenTypeIDs = Array(repeating: Int32(0), count: activeTokenCount)

        if activeTokenCount < sequenceLength {
            let padCount = sequenceLength - activeTokenCount
            inputIDs.append(contentsOf: Array(repeating: padTokenID, count: padCount))
            attentionMask.append(contentsOf: Array(repeating: 0, count: padCount))
            tokenTypeIDs.append(contentsOf: Array(repeating: 0, count: padCount))
        }

        return RampartEncodedInput(
            tokens: tokens,
            inputIDs: inputIDs,
            attentionMask: attentionMask,
            tokenTypeIDs: tokenTypeIDs,
            activeTokenCount: activeTokenCount,
            truncated: truncated
        )
    }

    func contentTokens(_ text: String) -> [RampartToken] {
        basicTokenize(text).flatMap { token in
            wordPieceTokenize(token)
        }
    }

    private func wordPieceTokenize(_ token: BasicToken) -> [RampartToken] {
        let characters = Array(token.normalized)
        guard !characters.isEmpty else { return [] }
        guard characters.count <= maxInputCharactersPerWord else {
            return [unknownRampartToken(for: token)]
        }

        var pieces: [RampartToken] = []
        var start = 0

        while start < characters.count {
            var end = characters.count
            var currentSubstring: String?

            while start < end {
                var substring = String(characters[start..<end])
                if start > 0 {
                    substring = "##" + substring
                }
                if vocabulary[substring] != nil {
                    currentSubstring = substring
                    break
                }
                end -= 1
            }

            guard let currentSubstring else {
                return [unknownRampartToken(for: token)]
            }

            let pieceStart = token.characterRanges[start].lowerBound
            let pieceEnd = token.characterRanges[end - 1].upperBound
            pieces.append(
                RampartToken(
                    text: currentSubstring,
                    id: vocabulary[currentSubstring]!,
                    range: pieceStart..<pieceEnd,
                    isSpecialToken: false
                )
            )
            start = end
        }

        return pieces
    }

    private func unknownRampartToken(for token: BasicToken) -> RampartToken {
        RampartToken(
            text: unknownToken,
            id: vocabulary[unknownToken]!,
            range: token.range,
            isSpecialToken: false
        )
    }
}

private struct BasicToken {
    let normalized: String
    let characterRanges: [Range<String.Index>]

    var range: Range<String.Index> {
        characterRanges.first!.lowerBound..<characterRanges.last!.upperBound
    }
}

private extension BertWordPieceTokenizer {
    func basicTokenize(_ text: String) -> [BasicToken] {
        var tokens: [BasicToken] = []
        var currentScalars = ""
        var currentRanges: [Range<String.Index>] = []

        func flushCurrent() {
            guard !currentScalars.isEmpty else { return }
            tokens.append(BasicToken(normalized: currentScalars, characterRanges: currentRanges))
            currentScalars = ""
            currentRanges = []
        }

        var index = text.startIndex
        while index < text.endIndex {
            let nextIndex = text.index(after: index)
            let character = text[index]

            if character.isBertWhitespace || character.isBertControl {
                flushCurrent()
                index = nextIndex
                continue
            }

            if character.isBertPunctuation {
                flushCurrent()
                let normalized = normalizeCharacter(character)
                if !normalized.isEmpty {
                    tokens.append(BasicToken(normalized: normalized, characterRanges: [index..<nextIndex]))
                }
                index = nextIndex
                continue
            }

            let normalized = normalizeCharacter(character)
            if normalized.isEmpty {
                flushCurrent()
            } else {
                for scalar in normalized.unicodeScalars {
                    currentScalars.unicodeScalars.append(scalar)
                    currentRanges.append(index..<nextIndex)
                }
            }

            index = nextIndex
        }

        flushCurrent()
        return tokens
    }

    func normalizeCharacter(_ character: Character) -> String {
        let lowercased = String(character).lowercased()
        let decomposed = lowercased.decomposedStringWithCanonicalMapping
        var normalized = ""
        for scalar in decomposed.unicodeScalars {
            guard scalar.properties.generalCategory != .nonspacingMark else {
                continue
            }
            normalized.unicodeScalars.append(scalar)
        }
        return normalized
    }
}

private extension Character {
    var isBertWhitespace: Bool {
        unicodeScalars.allSatisfy { scalar in
            scalar == "\t" || scalar == "\n" || scalar == "\r" || scalar == " " || scalar.properties.isWhitespace
        }
    }

    var isBertControl: Bool {
        unicodeScalars.contains { scalar in
            scalar.properties.generalCategory == .control && scalar != "\t" && scalar != "\n" && scalar != "\r"
        }
    }

    var isBertPunctuation: Bool {
        unicodeScalars.count == 1 && unicodeScalars.contains(where: { scalar in
            let value = scalar.value
            if (33...47).contains(value) || (58...64).contains(value) || (91...96).contains(value) || (123...126).contains(value) {
                return true
            }
            switch scalar.properties.generalCategory {
            case .connectorPunctuation,
                 .dashPunctuation,
                 .openPunctuation,
                 .closePunctuation,
                 .initialPunctuation,
                 .finalPunctuation,
                 .otherPunctuation:
                return true
            default:
                return false
            }
        })
    }
}

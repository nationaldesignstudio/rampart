import Foundation

public enum RampartDetectionPolicy {
    public static let defaultKeepLabels: Set<String> = ["CITY", "STATE", "ZIP_CODE"]

    public static func redactableDetections(
        from detections: [PIIDetection],
        in input: String,
        keepLabels: Set<String> = defaultKeepLabels
    ) -> [PIIDetection] {
        merge(detections, in: input)
            .filter { !keepLabels.contains($0.label) }
            .sorted { $0.range.lowerBound > $1.range.lowerBound }
    }

    public static func redactedText(
        for input: String,
        detections: [PIIDetection]
    ) -> String {
        var redacted = input
        for detection in detections.sorted(by: { $0.range.lowerBound > $1.range.lowerBound }) {
            redacted.replaceSubrange(detection.range, with: "[\(detection.label)]")
        }
        return redacted
    }

    static func merge(_ detections: [PIIDetection], in input: String) -> [PIIDetection] {
        let sorted = detections.sorted {
            if $0.range.lowerBound == $1.range.lowerBound {
                return input.distance(from: $0.range.lowerBound, to: $0.range.upperBound)
                    > input.distance(from: $1.range.lowerBound, to: $1.range.upperBound)
            }
            return $0.range.lowerBound < $1.range.lowerBound
        }

        var merged: [PIIDetection] = []
        for detection in sorted {
            guard let previous = merged.last,
                  previous.range.overlaps(detection.range)
            else {
                merged.append(detection)
                continue
            }

            let winner = preferred(previous, detection, in: input)
            let previousContainsDetection =
                previous.range.lowerBound <= detection.range.lowerBound
                && previous.range.upperBound >= detection.range.upperBound
            let detectionContainsPrevious =
                detection.range.lowerBound <= previous.range.lowerBound
                && detection.range.upperBound >= previous.range.upperBound

            if previousContainsDetection || detectionContainsPrevious {
                merged[merged.count - 1] = winner
            } else {
                let lowerBound = min(previous.range.lowerBound, detection.range.lowerBound)
                let upperBound = max(previous.range.upperBound, detection.range.upperBound)
                merged[merged.count - 1] = PIIDetection(
                    label: winner.label,
                    range: lowerBound..<upperBound,
                    text: String(input[lowerBound..<upperBound]),
                    source: winner.source,
                    score: winner.score
                )
            }
        }

        return merged
    }

    private static func preferred(
        _ lhs: PIIDetection,
        _ rhs: PIIDetection,
        in input: String
    ) -> PIIDetection {
        let lhsScore = lhs.score ?? 0
        let rhsScore = rhs.score ?? 0
        if lhsScore != rhsScore {
            return lhsScore > rhsScore ? lhs : rhs
        }

        let lhsLength = input.distance(from: lhs.range.lowerBound, to: lhs.range.upperBound)
        let rhsLength = input.distance(from: rhs.range.lowerBound, to: rhs.range.upperBound)
        if lhsLength != rhsLength {
            return lhsLength > rhsLength ? lhs : rhs
        }

        return lhs.source == .deterministic ? lhs : rhs
    }
}

enum RampartModelDetectionBuilder {
    private static let extendScore: Float = 0.15

    static func detections(
        from predictions: [TokenPrediction],
        in input: String,
        minimumScore: Float
    ) -> [PIIDetection] {
        var detections: [PIIDetection] = []
        var builder: ModelSpanBuilder?

        func flush() {
            guard let current = builder else {
                return
            }
            if let detection = current.makeDetection(in: input), detection.score ?? 0 >= extendScore {
                detections.append(detection)
            }
            builder = nil
        }

        for prediction in predictions {
            guard prediction.label != "O", let range = prediction.range else {
                flush()
                continue
            }

            let label = normalizedLabel(prediction.label)
            if var current = builder,
               current.canMerge(
                   label: label,
                   range: range,
                   token: prediction.token,
                   rawLabel: prediction.label,
                   in: input
               ) {
                current.merge(range: range, score: prediction.score)
                builder = current
            } else {
                flush()
                builder = ModelSpanBuilder(
                    label: label,
                    range: range,
                    scoreTotal: prediction.score,
                    tokenCount: 1
                )
            }
        }

        flush()
        return ModelSpanRepair.repair(detections, in: input, minimumScore: minimumScore)
    }

    private static func normalizedLabel(_ label: String) -> String {
        if label.hasPrefix("B-") || label.hasPrefix("I-") {
            return String(label.dropFirst(2))
        }
        return label
    }
}

private struct ModelSpanBuilder {
    var label: String
    var range: Range<String.Index>
    var scoreTotal: Float
    var tokenCount: Int

    func canMerge(
        label nextLabel: String,
        range nextRange: Range<String.Index>,
        token: String,
        rawLabel: String,
        in input: String
    ) -> Bool {
        guard label == nextLabel else {
            return false
        }
        if token.hasPrefix("##") || rawLabel.hasPrefix("I-") {
            return true
        }
        return range.overlaps(nextRange)
    }

    mutating func merge(range nextRange: Range<String.Index>, score nextScore: Float) {
        range = min(range.lowerBound, nextRange.lowerBound)..<max(range.upperBound, nextRange.upperBound)
        scoreTotal += nextScore
        tokenCount += 1
    }

    func makeDetection(in input: String) -> PIIDetection? {
        guard range.lowerBound < range.upperBound, tokenCount > 0 else {
            return nil
        }
        return PIIDetection(
            label: label,
            range: range,
            text: String(input[range]),
            source: .model,
            score: scoreTotal / Float(tokenCount)
        )
    }
}

private enum ModelSpanRepair {
    private static let personLabels: Set<String> = ["GIVEN_NAME", "SURNAME"]
    private static let maxIterations = 32

    static func repair(
        _ detections: [PIIDetection],
        in input: String,
        minimumScore: Float
    ) -> [PIIDetection] {
        var kept = detections.filter { ($0.score ?? 0) >= minimumScore }
        var candidates = detections.filter {
            let score = $0.score ?? 0
            return score >= 0.15 && score < minimumScore
        }

        var changed = true
        var iteration = 0
        while changed && iteration < maxIterations {
            changed = false
            iteration += 1

            for index in candidates.indices.reversed() {
                let candidate = candidates[index]
                if kept.contains(where: { canBridge(candidate, $0, in: input) }) {
                    kept.append(candidate)
                    candidates.remove(at: index)
                    changed = true
                }
            }

            let merged = mergeAdjacentConnectors(kept, in: input)
            if merged.map(\.range) != kept.map(\.range) {
                changed = true
            }
            kept = merged

            for index in kept.indices {
                let repaired = rescueCapitalizedParticles(kept[index], all: kept, selfIndex: index, in: input)
                if repaired.range != kept[index].range {
                    kept[index] = repaired
                    changed = true
                }
            }
        }

        return kept.sorted { $0.range.lowerBound < $1.range.lowerBound }
    }

    static func canBridge(
        label: String,
        left: Range<String.Index>,
        right: Range<String.Index>,
        in input: String
    ) -> Bool {
        guard label != "O", left.upperBound <= right.lowerBound else {
            return left.overlaps(right)
        }
        let gap = input[left.upperBound..<right.lowerBound]
        guard isConnector(gap) else {
            return false
        }
        if gap.contains("."), !isInitialCharacterEnding(left, in: input) {
            return false
        }
        return true
    }

    private static func canBridge(
        _ lhs: PIIDetection,
        _ rhs: PIIDetection,
        in input: String
    ) -> Bool {
        guard lhs.label == rhs.label else {
            return false
        }
        let left = lhs.range.lowerBound <= rhs.range.lowerBound ? lhs : rhs
        let right = left == lhs ? rhs : lhs
        return canBridge(label: left.label, left: left.range, right: right.range, in: input)
    }

    private static func mergeAdjacentConnectors(
        _ detections: [PIIDetection],
        in input: String
    ) -> [PIIDetection] {
        var merged: [PIIDetection] = []
        for detection in detections.sorted(by: { $0.range.lowerBound < $1.range.lowerBound }) {
            guard let previous = merged.last,
                  canBridge(previous, detection, in: input)
            else {
                merged.append(detection)
                continue
            }

            let lowerBound = min(previous.range.lowerBound, detection.range.lowerBound)
            let upperBound = max(previous.range.upperBound, detection.range.upperBound)
            let score = max(previous.score ?? 0, detection.score ?? 0)
            merged[merged.count - 1] = PIIDetection(
                label: previous.label,
                range: lowerBound..<upperBound,
                text: String(input[lowerBound..<upperBound]),
                source: .model,
                score: score
            )
        }
        return merged
    }

    private static func rescueCapitalizedParticles(
        _ detection: PIIDetection,
        all detections: [PIIDetection],
        selfIndex: Int,
        in input: String
    ) -> PIIDetection {
        guard personLabels.contains(detection.label) else {
            return detection
        }

        var leftBound = input.startIndex
        var rightBound = input.endIndex
        for (index, other) in detections.enumerated() where index != selfIndex {
            if other.range.upperBound <= detection.range.lowerBound,
               other.range.upperBound > leftBound {
                leftBound = other.range.upperBound
            }
            if other.range.lowerBound >= detection.range.upperBound,
               other.range.lowerBound < rightBound {
                rightBound = other.range.lowerBound
            }
        }

        var range = detection.range
        if let leftRange = capitalizedParticleBefore(range.lowerBound, leftBound: leftBound, in: input) {
            range = leftRange.lowerBound..<range.upperBound
        }
        if let rightRange = capitalizedParticleAfter(range.upperBound, rightBound: rightBound, in: input) {
            range = range.lowerBound..<rightRange.upperBound
        }

        guard range != detection.range else {
            return detection
        }
        return PIIDetection(
            label: detection.label,
            range: range,
            text: String(input[range]),
            source: detection.source,
            score: detection.score
        )
    }

    private static func capitalizedParticleBefore(
        _ index: String.Index,
        leftBound: String.Index,
        in input: String
    ) -> Range<String.Index>? {
        guard index > leftBound else {
            return nil
        }

        var connectorStart = index
        var cursor = index
        var connectorCount = 0
        while cursor > leftBound, connectorCount < 3 {
            let previous = input.index(before: cursor)
            guard isConnectorCharacter(input[previous]) else {
                break
            }
            connectorStart = previous
            cursor = previous
            connectorCount += 1
        }
        guard connectorCount > 0 else {
            return nil
        }

        var wordStart = connectorStart
        cursor = connectorStart
        var wordCount = 0
        while cursor > leftBound, wordCount < 4 {
            let previous = input.index(before: cursor)
            guard isNameParticleCharacter(input[previous]) else {
                break
            }
            wordStart = previous
            cursor = previous
            wordCount += 1
        }
        guard wordCount > 0,
              isUppercaseLetter(input[wordStart])
        else {
            return nil
        }

        let connector = input[connectorStart..<index]
        if connector.contains("."), wordCount != 1 {
            return nil
        }
        return wordStart..<index
    }

    private static func capitalizedParticleAfter(
        _ index: String.Index,
        rightBound: String.Index,
        in input: String
    ) -> Range<String.Index>? {
        guard index < rightBound else {
            return nil
        }

        var connectorEnd = index
        var cursor = index
        var connectorCount = 0
        while cursor < rightBound, connectorCount < 3 {
            guard isConnectorCharacter(input[cursor]) else {
                break
            }
            connectorEnd = input.index(after: cursor)
            cursor = connectorEnd
            connectorCount += 1
        }
        guard connectorCount > 0 else {
            return nil
        }

        let connector = input[index..<connectorEnd]
        guard !connector.contains("."),
              connectorEnd < rightBound,
              isUppercaseLetter(input[connectorEnd])
        else {
            return nil
        }

        var wordEnd = connectorEnd
        cursor = connectorEnd
        var wordCount = 0
        while cursor < rightBound, wordCount < 4, isNameParticleCharacter(input[cursor]) {
            wordEnd = input.index(after: cursor)
            cursor = wordEnd
            wordCount += 1
        }
        guard wordCount > 0 else {
            return nil
        }
        return index..<wordEnd
    }

    private static func isConnector<S: StringProtocol>(_ text: S) -> Bool {
        !text.isEmpty && text.allSatisfy(isConnectorCharacter)
    }

    private static func isConnectorCharacter(_ character: Character) -> Bool {
        character.isWhitespace || character == "'" || character == "\u{2019}" || character == "." || character == "-"
    }

    private static func isNameParticleCharacter(_ character: Character) -> Bool {
        isLetter(character) || character == "'" || character == "\u{2019}"
    }

    private static func isInitialCharacterEnding(_ range: Range<String.Index>, in input: String) -> Bool {
        guard range.lowerBound < range.upperBound else {
            return false
        }
        let lastIndex = input.index(before: range.upperBound)
        guard isUppercaseLetter(input[lastIndex]) else {
            return false
        }
        if lastIndex == input.startIndex {
            return true
        }
        let previousIndex = input.index(before: lastIndex)
        return !isLetter(input[previousIndex])
    }

    private static func isLetter(_ character: Character) -> Bool {
        character.unicodeScalars.allSatisfy { scalar in
            scalar.properties.generalCategory == .uppercaseLetter ||
                scalar.properties.generalCategory == .lowercaseLetter ||
                scalar.properties.generalCategory == .titlecaseLetter ||
                scalar.properties.generalCategory == .modifierLetter ||
                scalar.properties.generalCategory == .otherLetter ||
                scalar.properties.generalCategory == .nonspacingMark
        }
    }

    private static func isUppercaseLetter(_ character: Character) -> Bool {
        character.unicodeScalars.contains { scalar in
            scalar.properties.generalCategory == .uppercaseLetter
        }
    }
}

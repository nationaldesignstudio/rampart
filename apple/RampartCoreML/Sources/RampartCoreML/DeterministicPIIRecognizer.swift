import Foundation

public enum PIIDetectionSource: String, Sendable {
    case deterministic
    case model
}

public struct PIIDetection: Equatable, Sendable {
    public let label: String
    public let range: Range<String.Index>
    public let text: String
    public let source: PIIDetectionSource
    public let score: Float?

    public init(
        label: String,
        range: Range<String.Index>,
        text: String,
        source: PIIDetectionSource,
        score: Float? = nil
    ) {
        self.label = label
        self.range = range
        self.text = text
        self.source = source
        self.score = score
    }
}

public struct DeterministicPIIRecognizer: Sendable {
    public static let standard = DeterministicPIIRecognizer()

    public init() {}

    public func recognize(in text: String) -> [PIIDetection] {
        var detections: [PIIDetection] = []
        detections.append(contentsOf: recognizeDigitEntities(in: text))
        detections.append(contentsOf: recognizeTextEntities(in: text))
        return detections.sorted { $0.range.lowerBound < $1.range.lowerBound }
    }

    private func recognizeDigitEntities(in text: String) -> [PIIDetection] {
        extractDigitRuns(in: text).compactMap { run in
            if [16, 15, 14].contains(run.digits.count), isLuhnValid(run.digits) {
                return detection(label: "CREDIT_CARD", run: run, in: text)
            }
            if run.digits.count == 9, isValidSSN(run.digits) {
                return detection(label: "SSN", run: run, in: text)
            }
            return nil
        }
    }

    private func extractDigitRuns(in text: String) -> [DigitRun] {
        let regex = try! NSRegularExpression(pattern: #"\d(?:[ .-]?\d)*"#)
        let range = NSRange(text.startIndex..<text.endIndex, in: text)

        return regex.matches(in: text, range: range).compactMap { match in
            guard let matchRange = Range(match.range(at: 0), in: text) else {
                return nil
            }

            var digits = ""
            var indexes: [String.Index] = []
            for index in text[matchRange].indices {
                guard let digit = asciiDigit(at: index, in: text) else {
                    continue
                }
                digits.append(digit)
                indexes.append(index)
            }

            guard !digits.isEmpty else {
                return nil
            }
            return DigitRun(digits: digits, indexes: indexes)
        }
    }

    private func recognizeTextEntities(in text: String) -> [PIIDetection] {
        let rules: [(label: String, pattern: String)] = [
            ("EMAIL", #"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"#),
            ("URL", #"\bhttps?:\/\/[^\s<>"'\])}]+"#),
            ("URL", #"\bwww\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:\/[^\s<>"'\])}]*)?"#),
            ("IP_ADDRESS", #"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"#),
            (
                "IP_ADDRESS",
                #"(?<![:.\w])(?:(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}|(?:[0-9A-Fa-f]{1,4}:){1,7}:|(?:[0-9A-Fa-f]{1,4}:){1,6}:[0-9A-Fa-f]{1,4}|(?:[0-9A-Fa-f]{1,4}:){1,5}(?::[0-9A-Fa-f]{1,4}){1,2}|(?:[0-9A-Fa-f]{1,4}:){1,4}(?::[0-9A-Fa-f]{1,4}){1,3}|(?:[0-9A-Fa-f]{1,4}:){1,3}(?::[0-9A-Fa-f]{1,4}){1,4}|(?:[0-9A-Fa-f]{1,4}:){1,2}(?::[0-9A-Fa-f]{1,4}){1,5}|[0-9A-Fa-f]{1,4}:(?::[0-9A-Fa-f]{1,4}){1,6}|::(?:[0-9A-Fa-f]{1,4}:){0,6}[0-9A-Fa-f]{1,4})(?![:.\w])"#
            ),
            ("IP_ADDRESS", #"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"#)
        ]

        var detections: [PIIDetection] = []
        let range = NSRange(text.startIndex..<text.endIndex, in: text)

        for rule in rules {
            let regex = try! NSRegularExpression(pattern: rule.pattern)
            for match in regex.matches(in: text, range: range) {
                guard let matchRange = Range(match.range(at: 0), in: text) else {
                    continue
                }
                detections.append(
                    PIIDetection(
                        label: rule.label,
                        range: matchRange,
                        text: String(text[matchRange]),
                        source: .deterministic,
                        score: 1
                    )
                )
            }
        }

        return detections
    }

    private func detection(label: String, run: DigitRun, in text: String) -> PIIDetection? {
        guard let start = run.indexes.first, let lastDigit = run.indexes.last else {
            return nil
        }
        let end = text.index(after: lastDigit)
        let range = start..<end
        return PIIDetection(
            label: label,
            range: range,
            text: String(text[range]),
            source: .deterministic,
            score: 1
        )
    }

    private func isValidSSN(_ digits: String) -> Bool {
        guard digits.count == 9 else {
            return false
        }
        let area = String(digits.prefix(3))
        let groupStart = digits.index(digits.startIndex, offsetBy: 3)
        let groupEnd = digits.index(groupStart, offsetBy: 2)
        let group = String(digits[groupStart..<groupEnd])
        let serial = String(digits[groupEnd...])
        guard let areaNumber = Int(area) else {
            return false
        }
        guard areaNumber != 0, areaNumber != 666, areaNumber < 900 else {
            return false
        }
        guard group != "00", serial != "0000" else {
            return false
        }
        return true
    }

    private func isLuhnValid(_ digits: String) -> Bool {
        var sum = 0
        var shouldDouble = false

        for character in digits.reversed() {
            guard var value = character.wholeNumberValue else {
                return false
            }
            if shouldDouble {
                value *= 2
                if value > 9 {
                    value -= 9
                }
            }
            sum += value
            shouldDouble.toggle()
        }

        return sum % 10 == 0
    }

    private func asciiDigit(at index: String.Index, in text: String) -> Character? {
        let character = text[index]
        guard character.unicodeScalars.count == 1, let scalar = character.unicodeScalars.first else {
            return nil
        }
        guard (48...57).contains(scalar.value) else {
            return nil
        }
        return character
    }
}

private struct DigitRun {
    let digits: String
    let indexes: [String.Index]
}

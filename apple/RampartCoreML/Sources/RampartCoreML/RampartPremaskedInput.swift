import Foundation

struct RampartPremaskedInput: Sendable {
    let original: String
    let masked: String

    private let rawStart: [String.Index]
    private let rawEnd: [String.Index]

    init(original: String, detections: [PIIDetection]) {
        self.original = original

        let ordered = RampartDetectionPolicy.merge(detections, in: original)
            .sorted { $0.range.lowerBound < $1.range.lowerBound }

        var masked = ""
        var rawStart: [String.Index] = []
        var rawEnd: [String.Index] = []

        func copyVerbatim(from start: String.Index, to end: String.Index) {
            var index = start
            while index < end {
                let nextIndex = original.index(after: index)
                masked.append(original[index])
                rawStart.append(index)
                rawEnd.append(nextIndex)
                index = nextIndex
            }
        }

        var cursor = original.startIndex
        for detection in ordered {
            guard detection.range.lowerBound >= cursor else {
                continue
            }

            copyVerbatim(from: cursor, to: detection.range.lowerBound)

            for character in Self.sentinel(for: detection.label) {
                masked.append(character)
                rawStart.append(detection.range.lowerBound)
                rawEnd.append(detection.range.upperBound)
            }

            cursor = detection.range.upperBound
        }

        copyVerbatim(from: cursor, to: original.endIndex)

        self.masked = masked.replacingOccurrences(of: "-", with: " ")
        self.rawStart = rawStart
        self.rawEnd = rawEnd
    }

    func project(_ range: Range<String.Index>?, maskedStartOffset: Int = 0) -> Range<String.Index>? {
        guard let range else {
            return nil
        }

        let startOffset = maskedStartOffset + masked.distance(from: masked.startIndex, to: range.lowerBound)
        let endOffset = maskedStartOffset + masked.distance(from: masked.startIndex, to: range.upperBound)

        guard startOffset >= 0,
              endOffset > startOffset,
              startOffset < rawStart.count,
              endOffset <= rawEnd.count
        else {
            return nil
        }

        let lowerBound = rawStart[startOffset]
        let upperBound = rawEnd[endOffset - 1]
        guard lowerBound < upperBound else {
            return nil
        }
        return lowerBound..<upperBound
    }

    static func sentinel(for label: String) -> String {
        "[\(label)]"
    }

}

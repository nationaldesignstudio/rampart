import CoreML
import Foundation

public struct RampartProtectionResult: Equatable, Sendable {
    public let protectedText: String
    public let placeholders: [String]
    public let detections: [PIIDetection]

    public init(protectedText: String, placeholders: [String], detections: [PIIDetection] = []) {
        self.protectedText = protectedText
        self.placeholders = placeholders
        self.detections = detections
    }
}

final class RampartPlaceholderStore {
    private var forward: [String: String] = [:]
    private var reverse: [String: String] = [:]
    private var counters: [String: Int] = [:]
    private let keepLabels: Set<String>

    init(
        keepLabels: Set<String> = RampartDetectionPolicy.defaultKeepLabels
    ) {
        self.keepLabels = keepLabels
    }

    func placeholder(for label: String, value: String) -> String {
        let key = "\(label):\(Self.normalizedValue(value))"
        if let existing = forward[key] {
            return existing
        }

        let next = (counters[label] ?? 0) + 1
        counters[label] = next

        let token = "[\(label)_\(next)]"
        forward[key] = token
        reverse[token] = value
        return token
    }

    func protect(_ raw: String, detections: [PIIDetection]) -> RampartProtectionResult {
        let redactable = RampartDetectionPolicy.redactableDetections(
            from: detections,
            in: raw,
            keepLabels: keepLabels
        )
        let ordered = redactable.sorted { $0.range.lowerBound < $1.range.lowerBound }

        var output = ""
        var placeholders: [String] = []
        var cursor = raw.startIndex

        for detection in ordered {
            output.append(contentsOf: raw[cursor..<detection.range.lowerBound])
            let token = placeholder(for: detection.label, value: detection.text)
            output.append(token)
            placeholders.append(token)
            cursor = detection.range.upperBound
        }

        output.append(contentsOf: raw[cursor..<raw.endIndex])
        return RampartProtectionResult(
            protectedText: output,
            placeholders: placeholders,
            detections: ordered
        )
    }

    func reveal(_ text: String) -> String {
        Self.replacingPlaceholders(in: text) { token in
            reverse[token]
        }
    }

    private static func normalizedValue(_ value: String) -> String {
        value
            .lowercased()
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
    }

    fileprivate static func replacingPlaceholders(
        in text: String,
        resolve: (String) -> String?
    ) -> String {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = placeholderExpression.matches(in: text, range: range)
        guard !matches.isEmpty else {
            return text
        }

        var output = ""
        var cursor = text.startIndex

        for match in matches {
            guard let tokenRange = Range(match.range, in: text) else {
                continue
            }
            output.append(contentsOf: text[cursor..<tokenRange.lowerBound])
            let token = String(text[tokenRange])
            output.append(resolve(token) ?? token)
            cursor = tokenRange.upperBound
        }

        output.append(contentsOf: text[cursor..<text.endIndex])
        return output
    }

    private static let placeholderExpression = try! NSRegularExpression(
        pattern: #"\[[A-Z][A-Z_]*_\d+\]"#
    )
}

public final class RampartGuard {
    private let classifier: RampartCoreMLClassifier?
    private let deterministicRecognizer: DeterministicPIIRecognizer
    private let placeholderStore: RampartPlaceholderStore

    public init(
        classifier: RampartCoreMLClassifier? = nil,
        keepLabels: Set<String> = RampartDetectionPolicy.defaultKeepLabels,
        deterministicRecognizer: DeterministicPIIRecognizer = .standard
    ) {
        self.classifier = classifier
        self.deterministicRecognizer = deterministicRecognizer
        self.placeholderStore = RampartPlaceholderStore(keepLabels: keepLabels)
    }

    public convenience init(
        artifacts: RampartModelArtifacts,
        keepLabels: Set<String> = RampartDetectionPolicy.defaultKeepLabels,
        sequenceLength: Int = 512,
        configuration: MLModelConfiguration = MLModelConfiguration(),
        deterministicRecognizer: DeterministicPIIRecognizer = .standard,
        minimumModelScore: Float = 0.4,
        tokenWindowOverlap: Int = 64
    ) throws {
        let classifier = try RampartCoreMLClassifier(
            artifacts: artifacts,
            sequenceLength: sequenceLength,
            configuration: configuration,
            deterministicRecognizer: deterministicRecognizer,
            minimumModelScore: minimumModelScore,
            tokenWindowOverlap: tokenWindowOverlap
        )
        self.init(
            classifier: classifier,
            keepLabels: keepLabels,
            deterministicRecognizer: deterministicRecognizer
        )
    }

    public static func downloaded(
        to rootURL: URL = RampartModelArtifacts.defaultCacheRootURL(),
        from downloadURL: URL = RampartModelArtifacts.defaultDownloadURL,
        forceDownload: Bool = false,
        keepLabels: Set<String> = RampartDetectionPolicy.defaultKeepLabels,
        sequenceLength: Int = 512,
        configuration: MLModelConfiguration = MLModelConfiguration(),
        deterministicRecognizer: DeterministicPIIRecognizer = .standard,
        minimumModelScore: Float = 0.4,
        tokenWindowOverlap: Int = 64
    ) async throws -> RampartGuard {
        let classifier = try await RampartCoreMLClassifier.downloaded(
            to: rootURL,
            from: downloadURL,
            forceDownload: forceDownload,
            sequenceLength: sequenceLength,
            configuration: configuration,
            deterministicRecognizer: deterministicRecognizer,
            minimumModelScore: minimumModelScore,
            tokenWindowOverlap: tokenWindowOverlap
        )
        return RampartGuard(
            classifier: classifier,
            keepLabels: keepLabels,
            deterministicRecognizer: deterministicRecognizer
        )
    }

    public func protect(_ text: String) throws -> RampartProtectionResult {
        try placeholderStore.protect(text, detections: detect(in: text))
    }

    public func reveal(_ reply: String) -> String {
        placeholderStore.reveal(reply)
    }

    private func detect(in text: String) throws -> [PIIDetection] {
        guard let classifier else {
            return deterministicRecognizer.recognize(in: text)
        }

        let result = try classifier.classify(text)
        return result.deterministicDetections + result.modelDetections
    }
}

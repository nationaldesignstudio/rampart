import CoreML
import Foundation

public struct TokenPrediction: Sendable {
    public let token: String
    public let range: Range<String.Index>?
    public let label: String
    public let score: Float

    public init(token: String, range: Range<String.Index>?, label: String, score: Float) {
        self.token = token
        self.range = range
        self.label = label
        self.score = score
    }
}

public struct RampartClassification: Sendable {
    public let input: String
    public let modelInput: String
    public let encoded: RampartEncodedInput
    public let predictions: [TokenPrediction]
    public let deterministicDetections: [PIIDetection]
    public let modelDetections: [PIIDetection]
    public let redactableDetections: [PIIDetection]

    public init(
        input: String,
        modelInput: String? = nil,
        encoded: RampartEncodedInput,
        predictions: [TokenPrediction],
        deterministicDetections: [PIIDetection] = [],
        modelDetections: [PIIDetection] = [],
        redactableDetections: [PIIDetection] = []
    ) {
        self.input = input
        self.modelInput = modelInput ?? input
        self.encoded = encoded
        self.predictions = predictions
        self.deterministicDetections = deterministicDetections
        self.modelDetections = modelDetections
        self.redactableDetections = redactableDetections
    }
}

public enum RampartCoreMLError: Error, LocalizedError {
    case missingLabel(Int)
    case invalidInputCount(expected: Int, actual: Int)
    case missingOutput(String)
    case invalidOutputShape([NSNumber])
    case unsupportedOutputDataType(MLMultiArrayDataType)

    public var errorDescription: String? {
        switch self {
        case .missingLabel(let index):
            "Config is missing label id \(index)."
        case .invalidInputCount(let expected, let actual):
            "Expected \(expected) tensor values but found \(actual)."
        case .missingOutput(let name):
            "Core ML output \(name) is missing."
        case .invalidOutputShape(let shape):
            "Core ML logits output has invalid shape \(shape)."
        case .unsupportedOutputDataType(let dataType):
            "Unsupported Core ML logits data type \(dataType.rawValue)."
        }
    }
}

public final class RampartCoreMLClassifier {
    public let tokenizer: BertWordPieceTokenizer
    public let labels: [String]
    public let sequenceLength: Int
    public let deterministicRecognizer: DeterministicPIIRecognizer
    public let minimumModelScore: Float
    public let tokenWindowOverlap: Int

    private let model: MLModel

    public convenience init(
        modelURL: URL,
        vocabURL: URL,
        configURL: URL,
        sequenceLength: Int = 512,
        configuration: MLModelConfiguration = MLModelConfiguration(),
        deterministicRecognizer: DeterministicPIIRecognizer = .standard,
        minimumModelScore: Float = 0.4,
        tokenWindowOverlap: Int = 64
    ) throws {
        let tokenizer = try BertWordPieceTokenizer(vocabURL: vocabURL)
        let labels = try Self.loadLabels(configURL: configURL)
        try self.init(
            modelURL: modelURL,
            tokenizer: tokenizer,
            labels: labels,
            sequenceLength: sequenceLength,
            configuration: configuration,
            deterministicRecognizer: deterministicRecognizer,
            minimumModelScore: minimumModelScore,
            tokenWindowOverlap: tokenWindowOverlap
        )
    }

    public convenience init(
        artifacts: RampartModelArtifacts,
        sequenceLength: Int = 512,
        configuration: MLModelConfiguration = MLModelConfiguration(),
        deterministicRecognizer: DeterministicPIIRecognizer = .standard,
        minimumModelScore: Float = 0.4,
        tokenWindowOverlap: Int = 64
    ) throws {
        try self.init(
            modelURL: artifacts.modelURL,
            vocabURL: artifacts.vocabURL,
            configURL: artifacts.configURL,
            sequenceLength: sequenceLength,
            configuration: configuration,
            deterministicRecognizer: deterministicRecognizer,
            minimumModelScore: minimumModelScore,
            tokenWindowOverlap: tokenWindowOverlap
        )
    }

    public static func downloaded(
        to rootURL: URL = RampartModelArtifacts.defaultCacheRootURL(),
        from downloadURL: URL = RampartModelArtifacts.defaultDownloadURL,
        forceDownload: Bool = false,
        sequenceLength: Int = 512,
        configuration: MLModelConfiguration = MLModelConfiguration(),
        deterministicRecognizer: DeterministicPIIRecognizer = .standard,
        minimumModelScore: Float = 0.4,
        tokenWindowOverlap: Int = 64
    ) async throws -> RampartCoreMLClassifier {
        let artifacts = try await RampartModelArtifacts.downloadIfNeeded(
            to: rootURL,
            from: downloadURL,
            forceDownload: forceDownload
        )
        return try RampartCoreMLClassifier(
            artifacts: artifacts,
            sequenceLength: sequenceLength,
            configuration: configuration,
            deterministicRecognizer: deterministicRecognizer,
            minimumModelScore: minimumModelScore,
            tokenWindowOverlap: tokenWindowOverlap
        )
    }

    public init(
        modelURL: URL,
        tokenizer: BertWordPieceTokenizer,
        labels: [String],
        sequenceLength: Int = 512,
        configuration: MLModelConfiguration = MLModelConfiguration(),
        deterministicRecognizer: DeterministicPIIRecognizer = .standard,
        minimumModelScore: Float = 0.4,
        tokenWindowOverlap: Int = 64
    ) throws {
        self.model = try Self.loadModel(modelURL: modelURL, configuration: configuration)
        self.tokenizer = tokenizer
        self.labels = labels
        self.sequenceLength = sequenceLength
        self.deterministicRecognizer = deterministicRecognizer
        self.minimumModelScore = minimumModelScore
        self.tokenWindowOverlap = tokenWindowOverlap
    }

    public func classify(_ text: String) throws -> RampartClassification {
        let deterministicDetections = deterministicRecognizer.recognize(in: text)
        let premasked = RampartPremaskedInput(original: text, detections: deterministicDetections)
        let predictionResult = try predictWindows(premasked)
        let predictions = predictionResult.predictions
        let modelDetections = RampartModelDetectionBuilder.detections(
            from: predictions,
            in: text,
            minimumScore: minimumModelScore
        )
        let redactableDetections = RampartDetectionPolicy.redactableDetections(
            from: deterministicDetections + modelDetections,
            in: text
        )
        return RampartClassification(
            input: text,
            modelInput: premasked.masked,
            encoded: predictionResult.firstEncoded,
            predictions: predictions,
            deterministicDetections: deterministicDetections,
            modelDetections: modelDetections,
            redactableDetections: redactableDetections
        )
    }

    private static func loadModel(
        modelURL: URL,
        configuration: MLModelConfiguration
    ) throws -> MLModel {
        if modelURL.pathExtension == "mlmodelc" {
            return try MLModel(contentsOf: modelURL, configuration: configuration)
        }
        let compiledURL = try MLModel.compileModel(at: modelURL)
        return try MLModel(contentsOf: compiledURL, configuration: configuration)
    }

    private static func loadLabels(configURL: URL) throws -> [String] {
        let data = try Data(contentsOf: configURL)
        let object = try JSONSerialization.jsonObject(with: data)
        guard
            let dictionary = object as? [String: Any],
            let id2Label = dictionary["id2label"] as? [String: String]
        else {
            return []
        }

        let count = id2Label.count
        return try (0..<count).map { index in
            guard let label = id2Label[String(index)] else {
                throw RampartCoreMLError.missingLabel(index)
            }
            return label
        }
    }

    private func predict(_ encoded: RampartEncodedInput) throws -> MLMultiArray {
        let inputIDs = try multiArray(encoded.inputIDs)
        let attentionMask = try multiArray(encoded.attentionMask)
        let tokenTypeIDs = try multiArray(encoded.tokenTypeIDs)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputIDs,
            "attention_mask": attentionMask,
            "token_type_ids": tokenTypeIDs
        ])
        let prediction = try model.prediction(from: provider)

        guard let output = prediction.featureValue(for: "logits")?.multiArrayValue else {
            throw RampartCoreMLError.missingOutput("logits")
        }
        return output
    }

    private func predictWindows(
        _ premasked: RampartPremaskedInput
    ) throws -> (firstEncoded: RampartEncodedInput, predictions: [TokenPrediction]) {
        let windows = modelWindows(for: premasked.masked)
        var firstEncoded: RampartEncodedInput?
        var predictions: [TokenPrediction] = []

        for window in windows {
            let encoded = tokenizer.encode(window.text, sequenceLength: sequenceLength)
            if firstEncoded == nil {
                firstEncoded = encoded
            }
            let output = try predict(encoded)
            let windowPredictions = try decodePredictions(
                from: output,
                encoded: encoded,
                premasked: premasked,
                maskedStartOffset: window.maskedStartOffset
            )
            predictions.append(contentsOf: windowPredictions)
        }

        if let firstEncoded {
            return (firstEncoded, predictions)
        }

        let emptyEncoded = tokenizer.encode(premasked.masked, sequenceLength: sequenceLength)
        return (emptyEncoded, [])
    }

    private func modelWindows(for text: String) -> [ModelWindow] {
        let contentTokens = tokenizer.contentTokens(text).filter { $0.range != nil }
        let budget = max(1, sequenceLength - 2 - 10)
        guard contentTokens.count > budget else {
            return [ModelWindow(text: text, maskedStartOffset: 0)]
        }

        let overlap = min(max(0, tokenWindowOverlap), max(0, budget - 1))
        var windows: [ModelWindow] = []
        var startTokenIndex = 0

        while startTokenIndex < contentTokens.count {
            var endTokenIndex = min(startTokenIndex + budget, contentTokens.count)
            if endTokenIndex < contentTokens.count {
                while endTokenIndex > startTokenIndex + 1,
                      contentTokens[endTokenIndex].text.hasPrefix("##") {
                    endTokenIndex -= 1
                }
            }

            guard let startRange = contentTokens[startTokenIndex].range,
                  let endRange = contentTokens[endTokenIndex - 1].range
            else {
                break
            }

            let windowRange = startRange.lowerBound..<endRange.upperBound
            let maskedStartOffset = text.distance(from: text.startIndex, to: windowRange.lowerBound)
            windows.append(
                ModelWindow(
                    text: String(text[windowRange]),
                    maskedStartOffset: maskedStartOffset
                )
            )

            if endTokenIndex == contentTokens.count {
                break
            }

            var nextStartTokenIndex = max(startTokenIndex + 1, endTokenIndex - overlap)
            while nextStartTokenIndex > startTokenIndex + 1,
                  contentTokens[nextStartTokenIndex].text.hasPrefix("##") {
                nextStartTokenIndex -= 1
            }
            startTokenIndex = nextStartTokenIndex
        }

        return windows
    }

    private func multiArray(_ values: [Int32]) throws -> MLMultiArray {
        guard values.count == sequenceLength else {
            throw RampartCoreMLError.invalidInputCount(expected: sequenceLength, actual: values.count)
        }

        let array = try MLMultiArray(shape: [1, NSNumber(value: sequenceLength)], dataType: .int32)
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        for index in values.indices {
            pointer[index] = values[index]
        }
        return array
    }

    private func decodePredictions(
        from logits: MLMultiArray,
        encoded: RampartEncodedInput,
        premasked: RampartPremaskedInput,
        maskedStartOffset: Int = 0
    ) throws -> [TokenPrediction] {
        guard logits.shape.count == 3,
              logits.shape[0].intValue == 1,
              logits.shape[1].intValue == sequenceLength,
              logits.shape[2].intValue == labels.count
        else {
            throw RampartCoreMLError.invalidOutputShape(logits.shape)
        }

        var predictions: [TokenPrediction] = []
        predictions.reserveCapacity(encoded.activeTokenCount)

        for tokenIndex in 0..<encoded.activeTokenCount {
            let token = encoded.tokens[tokenIndex]
            let (labelIndex, score) = try bestLabel(in: logits, tokenIndex: tokenIndex)
            predictions.append(
                TokenPrediction(
                    token: token.text,
                    range: premasked.project(token.range, maskedStartOffset: maskedStartOffset),
                    label: labels[labelIndex],
                    score: score
                )
            )
        }

        return predictions
    }

    private func bestLabel(in logits: MLMultiArray, tokenIndex: Int) throws -> (Int, Float) {
        let labelCount = labels.count
        var maxIndex = 0
        var maxLogit = -Float.infinity
        var values = Array(repeating: Float(0), count: labelCount)

        for labelIndex in 0..<labelCount {
            let value = try logitValue(logits, tokenIndex: tokenIndex, labelIndex: labelIndex)
            values[labelIndex] = value
            if value > maxLogit {
                maxLogit = value
                maxIndex = labelIndex
            }
        }

        var denominator: Float = 0
        for value in values {
            denominator += expf(value - maxLogit)
        }
        let score = expf(values[maxIndex] - maxLogit) / denominator
        return (maxIndex, score)
    }

    private func logitValue(_ logits: MLMultiArray, tokenIndex: Int, labelIndex: Int) throws -> Float {
        let flatIndex = tokenIndex * logits.strides[1].intValue + labelIndex * logits.strides[2].intValue
        switch logits.dataType {
        case .float32:
            return logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)[flatIndex]
        case .float16:
            return logits[[0, NSNumber(value: tokenIndex), NSNumber(value: labelIndex)]].floatValue
        case .double:
            return Float(logits.dataPointer.bindMemory(to: Double.self, capacity: logits.count)[flatIndex])
        default:
            throw RampartCoreMLError.unsupportedOutputDataType(logits.dataType)
        }
    }
}

private struct ModelWindow {
    let text: String
    let maskedStartOffset: Int
}

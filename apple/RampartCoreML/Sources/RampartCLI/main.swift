import Foundation
import RampartCoreML

struct CLIOptions {
    static let defaultModelPath = "artifacts/RampartTokenClassifier.mlpackage"
    static let defaultVocabPath = "artifacts/rampart-hf/vocab.txt"
    static let defaultConfigPath = "artifacts/rampart-hf/config.json"

    var textParts: [String] = []
    var showAllTokens = false
    var modelPath = defaultModelPath
    var vocabPath = defaultVocabPath
    var configPath = defaultConfigPath

    var text: String {
        textParts.joined(separator: " ")
    }

    var usesDefaultArtifactPaths: Bool {
        modelPath == Self.defaultModelPath &&
            vocabPath == Self.defaultVocabPath &&
            configPath == Self.defaultConfigPath
    }
}

func usage() -> String {
    """
    Usage:
      swift run RampartCLI -- "text to protect"
      swift run RampartCLI -- --all "text to protect"

    Options:
      --all              Print O-label tokens too.
      --model PATH       Core ML .mlpackage or compiled .mlmodelc path.
      --vocab PATH       Rampart vocab.txt path.
      --config PATH      Rampart config.json path.
    """
}

func parseOptions(_ arguments: [String]) throws -> CLIOptions {
    var options = CLIOptions()
    var index = 0

    while index < arguments.count {
        let argument = arguments[index]
        switch argument {
        case "--":
            break
        case "--help", "-h":
            print(usage())
            exit(0)
        case "--all":
            options.showAllTokens = true
        case "--model", "--vocab", "--config":
            let valueIndex = index + 1
            guard valueIndex < arguments.count else {
                throw CLIError.missingValue(argument)
            }
            let value = arguments[valueIndex]
            switch argument {
            case "--model":
                options.modelPath = value
            case "--vocab":
                options.vocabPath = value
            case "--config":
                options.configPath = value
            default:
                break
            }
            index += 1
        default:
            options.textParts.append(argument)
        }
        index += 1
    }

    guard !options.text.isEmpty else {
        throw CLIError.missingText
    }

    return options
}

enum CLIError: Error, LocalizedError {
    case missingText
    case missingValue(String)

    var errorDescription: String? {
        switch self {
        case .missingText:
            "Missing text to protect.\n\n\(usage())"
        case .missingValue(let flag):
            "Missing value for \(flag)."
        }
    }
}

func integerOffsets(for range: Range<String.Index>, in text: String) -> (Int, Int) {
    let start = text.distance(from: text.startIndex, to: range.lowerBound)
    let end = text.distance(from: text.startIndex, to: range.upperBound)
    return (start, end)
}

func padded(_ value: String, _ width: Int) -> String {
    if value.count >= width {
        return value
    }
    return value + String(repeating: " ", count: width - value.count)
}

func offsetText(for range: Range<String.Index>, in text: String) -> String {
    let offsets = integerOffsets(for: range, in: text)
    return "[\(offsets.0),\(offsets.1))"
}

func defaultArtifactsExist(for options: CLIOptions) -> Bool {
    let fileManager = FileManager.default
    return fileManager.fileExists(atPath: options.modelPath) &&
        fileManager.fileExists(atPath: options.vocabPath) &&
        fileManager.fileExists(atPath: options.configPath)
}

do {
    let options = try parseOptions(Array(CommandLine.arguments.dropFirst()))
    let classifier: RampartCoreMLClassifier
    if options.usesDefaultArtifactPaths && !defaultArtifactsExist(for: options) {
        print("Model artifacts not found. Downloading Rampart Core ML artifacts...")
        let repositoryRoot = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true
        )
        classifier = try await RampartCoreMLClassifier.downloaded(to: repositoryRoot)
    } else {
        classifier = try RampartCoreMLClassifier(
            modelURL: URL(fileURLWithPath: options.modelPath),
            vocabURL: URL(fileURLWithPath: options.vocabPath),
            configURL: URL(fileURLWithPath: options.configPath)
        )
    }
    let rampartGuard = RampartGuard(classifier: classifier)
    let protected = try rampartGuard.protect(options.text)
    let result = try classifier.classify(options.text)

    print("Input: \(options.text)")
    if result.modelInput != result.input {
        print("Model input: \(result.modelInput)")
    }
    print("Active tokens: \(result.encoded.activeTokenCount)")
    print("Truncated: \(result.encoded.truncated)")
    print("")

    if result.deterministicDetections.isEmpty {
        print("Deterministic matches: none")
    } else {
        print("Deterministic matches:")
        for detection in result.deterministicDetections {
            print(
                "  \(padded(detection.label, 14)) \(padded(offsetText(for: detection.range, in: options.text), 12)) \(detection.text)"
            )
        }
    }
    print("")

    if protected.detections.isEmpty {
        print("Redactable detections: none")
    } else {
        print("Redactable detections:")
        for detection in protected.detections.reversed() {
            let scoreText = detection.score.map { String(format: "%.4f", $0) } ?? "-"
            print(
                "  \(padded(detection.label, 18)) \(padded(detection.source.rawValue, 13)) \(padded(offsetText(for: detection.range, in: options.text), 12)) score=\(padded(scoreText, 7)) \(detection.text)"
            )
        }
    }
    print("Protected: \(protected.protectedText)")
    if !protected.placeholders.isEmpty {
        print("Placeholders: \(protected.placeholders.joined(separator: ", "))")
    }
    print("")

    let rows = result.predictions.enumerated().filter { _, prediction in
        options.showAllTokens || prediction.label != "O"
    }

    if rows.isEmpty {
        print("No non-O model predictions.")
    } else {
        print("Model predictions:")
        for (index, prediction) in rows {
            let offsetsDisplay: String
            let spanText: String
            if let range = prediction.range {
                offsetsDisplay = offsetText(for: range, in: options.text)
                spanText = String(options.text[range])
            } else {
                offsetsDisplay = "[-,-)"
                spanText = ""
            }

            let indexText = String(format: "%3d", index)
            let scoreText = String(format: "%.4f", prediction.score)
            print("\(indexText)  \(padded(prediction.token, 14)) \(padded(offsetsDisplay, 12)) \(padded(prediction.label, 18)) score=\(scoreText)  \(spanText)")
        }
    }
} catch {
    fputs("error: \(error.localizedDescription)\n", stderr)
    exit(1)
}

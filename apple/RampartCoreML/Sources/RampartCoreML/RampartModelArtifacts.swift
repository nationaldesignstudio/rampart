import Foundation
import ZIPFoundation

public struct RampartModelArtifacts: Sendable {
    public static let releaseTag = "0.1.3"
    public static let assetName = "rampart-coreml-artifacts.zip"
    public static let defaultDownloadURL = URL(
        string: "https://github.com/nationaldesignstudio/rampart/releases/download/\(releaseTag)/\(assetName)"
    )!

    public let rootURL: URL
    public let modelURL: URL
    public let vocabURL: URL
    public let configURL: URL

    public init(rootURL: URL) throws {
        self.rootURL = rootURL
        self.modelURL = rootURL.appendingPathComponent(
            "artifacts/RampartTokenClassifier.mlpackage",
            isDirectory: true
        )
        self.vocabURL = rootURL.appendingPathComponent("artifacts/rampart-hf/vocab.txt")
        self.configURL = rootURL.appendingPathComponent("artifacts/rampart-hf/config.json")
        try Self.validate(modelURL: modelURL, vocabURL: vocabURL, configURL: configURL)
    }

    public static func local(at rootURL: URL) throws -> RampartModelArtifacts {
        try RampartModelArtifacts(rootURL: rootURL)
    }

    public static func downloadIfNeeded(
        to rootURL: URL = defaultCacheRootURL(),
        from downloadURL: URL = defaultDownloadURL,
        forceDownload: Bool = false
    ) async throws -> RampartModelArtifacts {
        if !forceDownload, let artifacts = try? RampartModelArtifacts(rootURL: rootURL) {
            return artifacts
        }

        let fileManager = FileManager.default
        let temporaryRootURL = fileManager.temporaryDirectory
            .appendingPathComponent("RampartCoreML-\(UUID().uuidString)", isDirectory: true)
        let archiveURL = temporaryRootURL.appendingPathComponent(assetName)
        let extractionURL = temporaryRootURL.appendingPathComponent("extracted", isDirectory: true)

        try fileManager.createDirectory(at: temporaryRootURL, withIntermediateDirectories: true)
        defer {
            try? fileManager.removeItem(at: temporaryRootURL)
        }

        try await downloadArchive(from: downloadURL, to: archiveURL)
        try fileManager.createDirectory(at: extractionURL, withIntermediateDirectories: true)
        try fileManager.unzipItem(at: archiveURL, to: extractionURL)

        _ = try RampartModelArtifacts(rootURL: extractionURL)

        let parentURL = rootURL.deletingLastPathComponent()
        try fileManager.createDirectory(at: parentURL, withIntermediateDirectories: true)
        if fileManager.fileExists(atPath: rootURL.path) {
            try fileManager.removeItem(at: rootURL)
        }
        try fileManager.moveItem(at: extractionURL, to: rootURL)

        return try RampartModelArtifacts(rootURL: rootURL)
    }

    public static func defaultCacheRootURL() -> URL {
        let fileManager = FileManager.default
        let baseURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.temporaryDirectory
        return baseURL
            .appendingPathComponent("RampartCoreML", isDirectory: true)
            .appendingPathComponent(releaseTag, isDirectory: true)
    }

    private static func downloadArchive(from sourceURL: URL, to destinationURL: URL) async throws {
        let fileManager = FileManager.default
        if sourceURL.isFileURL {
            try fileManager.copyItem(at: sourceURL, to: destinationURL)
            return
        }

        let (temporaryURL, response) = try await URLSession.shared.download(from: sourceURL)
        if let httpResponse = response as? HTTPURLResponse,
           !(200..<300).contains(httpResponse.statusCode) {
            throw RampartModelArtifactsError.downloadFailed(statusCode: httpResponse.statusCode)
        }
        try fileManager.moveItem(at: temporaryURL, to: destinationURL)
    }

    private static func validate(modelURL: URL, vocabURL: URL, configURL: URL) throws {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: modelURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            throw RampartModelArtifactsError.missingArtifact(modelURL.path)
        }

        for url in [vocabURL, configURL] {
            guard fileManager.fileExists(atPath: url.path) else {
                throw RampartModelArtifactsError.missingArtifact(url.path)
            }
        }
    }
}

public enum RampartModelArtifactsError: Error, LocalizedError {
    case missingArtifact(String)
    case downloadFailed(statusCode: Int)

    public var errorDescription: String? {
        switch self {
        case .missingArtifact(let path):
            "Missing Rampart model artifact at \(path)."
        case .downloadFailed(let statusCode):
            "Rampart model artifact download failed with HTTP status \(statusCode)."
        }
    }
}

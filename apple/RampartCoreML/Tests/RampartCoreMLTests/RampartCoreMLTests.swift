import XCTest
import ZIPFoundation
@testable import RampartCoreML

final class RampartCoreMLTests: XCTestCase {
    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private var modelDir: URL {
        repoRoot.appendingPathComponent("artifacts/rampart-hf", isDirectory: true)
    }

    private var upstreamRepoRoot: URL {
        repoRoot
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private var upstreamModelDir: URL {
        upstreamRepoRoot.appendingPathComponent("model", isDirectory: true)
    }

    private func requireVocabURL() throws -> URL {
        for vocabURL in [
            modelDir.appendingPathComponent("vocab.txt"),
            upstreamModelDir.appendingPathComponent("vocab.txt")
        ] {
            guard FileManager.default.fileExists(atPath: vocabURL.path) else {
                continue
            }
            let contents = try String(contentsOf: vocabURL, encoding: .utf8)
            if contents.contains("[PAD]") {
                return vocabURL
            }
        }

        throw XCTSkip(
            """
            Run scripts/download_model.sh or scripts/convert_rampart_to_coreml.py \
            before tokenizer fixture tests.
            """
        )
    }

    private func requireArtifacts() throws -> RampartModelArtifacts {
        do {
            return try RampartModelArtifacts.local(at: repoRoot)
        } catch {
            throw XCTSkip(
                """
                Run scripts/download_model.sh or scripts/convert_rampart_to_coreml.py \
                before model integration tests.
                """
            )
        }
    }

    func testLocalArtifactsResolveExpectedPaths() throws {
        let artifacts = try requireArtifacts()

        XCTAssertEqual(
            artifacts.modelURL.path,
            repoRoot.appendingPathComponent("artifacts/RampartTokenClassifier.mlpackage").path
        )
        XCTAssertEqual(
            artifacts.vocabURL.path,
            repoRoot.appendingPathComponent("artifacts/rampart-hf/vocab.txt").path
        )
        XCTAssertEqual(
            artifacts.configURL.path,
            repoRoot.appendingPathComponent("artifacts/rampart-hf/config.json").path
        )
    }

    func testDownloadIfNeededExtractsReleaseArchive() async throws {
        let fileManager = FileManager.default
        let temporaryRoot = fileManager.temporaryDirectory
            .appendingPathComponent("RampartCoreMLTests-\(UUID().uuidString)", isDirectory: true)
        let sourceRoot = temporaryRoot.appendingPathComponent("source", isDirectory: true)
        let cacheRoot = temporaryRoot.appendingPathComponent("cache", isDirectory: true)
        let archiveURL = temporaryRoot.appendingPathComponent("artifacts.zip")

        defer {
            try? fileManager.removeItem(at: temporaryRoot)
        }

        try fileManager.createDirectory(
            at: sourceRoot.appendingPathComponent("artifacts/RampartTokenClassifier.mlpackage", isDirectory: true),
            withIntermediateDirectories: true
        )
        let modelManifest = sourceRoot.appendingPathComponent("artifacts/RampartTokenClassifier.mlpackage/Manifest.json")
        try "{}".write(to: modelManifest, atomically: true, encoding: .utf8)
        let hfRoot = sourceRoot.appendingPathComponent("artifacts/rampart-hf", isDirectory: true)
        try fileManager.createDirectory(at: hfRoot, withIntermediateDirectories: true)
        try "[PAD]\n[UNK]\n".write(to: hfRoot.appendingPathComponent("vocab.txt"), atomically: true, encoding: .utf8)
        try #"{"id2label":{"0":"O"}}"#.write(to: hfRoot.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        try fileManager.zipItem(at: sourceRoot, to: archiveURL, shouldKeepParent: false)

        let artifacts = try await RampartModelArtifacts.downloadIfNeeded(
            to: cacheRoot,
            from: archiveURL
        )

        XCTAssertEqual(artifacts.rootURL.path, cacheRoot.path)
        XCTAssertTrue(fileManager.fileExists(atPath: artifacts.modelURL.path))
        XCTAssertTrue(fileManager.fileExists(atPath: artifacts.vocabURL.path))
        XCTAssertTrue(fileManager.fileExists(atPath: artifacts.configURL.path))
    }

    func testTokenizerMatchesReferenceFixture() throws {
        let tokenizer = try BertWordPieceTokenizer(vocabURL: requireVocabURL())

        let encoded = tokenizer.encode(
            "My name is Clara and I live in Berkeley, California.",
            sequenceLength: 512
        )

        XCTAssertEqual(
            encoded.tokens.map(\.text),
            [
                "[CLS]", "my", "name", "is", "clara", "and", "i", "live", "in",
                "be", "##rke", "##ley", ",", "california", ".", "[SEP]"
            ]
        )
        XCTAssertEqual(encoded.inputIDs.prefix(16), [
            2, 15545, 921, 15941, 16934, 2550, 778, 1524, 8467,
            15427, 1594, 190, 8381, 13925, 7148, 3
        ])
        XCTAssertEqual(encoded.activeTokenCount, 16)
        XCTAssertFalse(encoded.truncated)
    }

    func testTokenizerNormalizesAccentsAndKeepsSourceRanges() throws {
        let tokenizer = try BertWordPieceTokenizer(vocabURL: requireVocabURL())

        let text = "josé müller moved from berlin to lisbon."
        let encoded = tokenizer.encode(text, sequenceLength: 512)

        XCTAssertEqual(
            encoded.tokens.prefix(10).map(\.text),
            ["[CLS]", "jose", "muller", "moved", "from", "berlin", "to", "li", "##sb", "##on"]
        )

        let jose = try XCTUnwrap(encoded.tokens[1].range)
        let muller = try XCTUnwrap(encoded.tokens[2].range)
        XCTAssertEqual(String(text[jose]), "josé")
        XCTAssertEqual(String(text[muller]), "müller")
    }

    func testClassifierLoadsModelAndPredictsExpectedLabels() throws {
        let classifier = try RampartCoreMLClassifier(artifacts: requireArtifacts())
        let text = "Alex Rivera lives at 221B Baker Street Apt 4 in London."
        let result = try classifier.classify(text)

        XCTAssertEqual(result.encoded.activeTokenCount, 15)
        XCTAssertEqual(result.predictions[1].token, "alex")
        XCTAssertEqual(result.predictions[1].label, "B-GIVEN_NAME")
        XCTAssertEqual(result.predictions[2].token, "rivera")
        XCTAssertEqual(result.predictions[2].label, "B-SURNAME")
        XCTAssertEqual(result.predictions[5].label, "B-BUILDING_NUMBER")
        XCTAssertEqual(result.predictions[7].label, "B-STREET_NAME")
        XCTAssertEqual(result.predictions[8].label, "I-STREET_NAME")
        XCTAssertEqual(result.predictions[12].label, "B-CITY")

        let alexRange = try XCTUnwrap(result.predictions[1].range)
        let londonRange = try XCTUnwrap(result.predictions[12].range)
        XCTAssertEqual(String(text[alexRange]), "Alex")
        XCTAssertEqual(String(text[londonRange]), "London")
        XCTAssertGreaterThan(result.predictions[1].score, 0)
        XCTAssertLessThanOrEqual(result.predictions[1].score, 1)
    }

    func testDeterministicRecognizerFindsStructuredPII() {
        let text = "my name is nick and my ssn is 111-11-1111"
        let detections = DeterministicPIIRecognizer.standard.recognize(in: text)

        XCTAssertEqual(detections.count, 1)
        XCTAssertEqual(detections[0].label, "SSN")
        XCTAssertEqual(detections[0].text, "111-11-1111")
        XCTAssertEqual(String(text[detections[0].range]), "111-11-1111")
    }

    func testDeterministicRecognizerFindsCreditCardEmailURLAndIP() {
        let text = """
        Card 4111 1111 1111 1111, email alex+housing@sub.example.gov, \
        visit https://example.com/private, ip 10.0.0.1.
        """
        let detections = DeterministicPIIRecognizer.standard.recognize(in: text)

        XCTAssertEqual(detections.map(\.label), ["CREDIT_CARD", "EMAIL", "URL", "IP_ADDRESS"])
        XCTAssertEqual(detections.map(\.text), [
            "4111 1111 1111 1111",
            "alex+housing@sub.example.gov",
            "https://example.com/private,",
            "10.0.0.1"
        ])
    }

    func testDeterministicRecognizerFindsIPv6AndMACAddress() {
        let text = "IPv6 2001:db8::1 and MAC 00:1B:44:11:3A:B7"
        let detections = DeterministicPIIRecognizer.standard.recognize(in: text)

        XCTAssertEqual(detections.map(\.label), ["IP_ADDRESS", "IP_ADDRESS"])
        XCTAssertEqual(detections.map(\.text), ["2001:db8::1", "00:1B:44:11:3A:B7"])
    }

    func testDeterministicRecognizerRejectsInvalidStructuredDigits() {
        let text = """
        phone-like 111-111-1111, invalid ssn 000-12-3456, \
        invalid card 1234 5678 1234 5678
        """
        let detections = DeterministicPIIRecognizer.standard.recognize(in: text)

        XCTAssertTrue(detections.isEmpty)
    }

    func testDeterministicRecognizerDoesNotTreatPhoneAsSSNWithoutKeyword() {
        let text = "call me at 415-555-0199"
        let detections = DeterministicPIIRecognizer.standard.recognize(in: text)

        XCTAssertTrue(detections.isEmpty)
    }

    func testPremaskedInputUsesRampartSentinelsAndProjectsRanges() throws {
        let text = "Alex-Rivera has SSN 111-11-1111 and email nick@example.com."
        let detections = DeterministicPIIRecognizer.standard.recognize(in: text)
        let premasked = RampartPremaskedInput(original: text, detections: detections)

        XCTAssertEqual(
            premasked.masked,
            "Alex Rivera has SSN [SSN] and email [EMAIL]."
        )

        let riveraMaskedRange = try XCTUnwrap(premasked.masked.range(of: "Rivera"))
        let riveraProjectedRange = try XCTUnwrap(premasked.project(riveraMaskedRange))
        XCTAssertEqual(String(text[riveraProjectedRange]), "Rivera")

        let ssnMaskedRange = try XCTUnwrap(premasked.masked.range(of: "[SSN]"))
        let ssnProjectedRange = try XCTUnwrap(premasked.project(ssnMaskedRange))
        XCTAssertEqual(String(text[ssnProjectedRange]), "111-11-1111")

        let emailMaskedRange = try XCTUnwrap(premasked.masked.range(of: "[EMAIL]"))
        let emailProjectedRange = try XCTUnwrap(premasked.project(emailMaskedRange))
        XCTAssertEqual(String(text[emailProjectedRange]), "nick@example.com")
    }

    func testModelDetectionBuilderMergesBIOPredictions() throws {
        let text = "Nick lives in San Francisco."
        let nickRange = try XCTUnwrap(text.range(of: "Nick"))
        let sanRange = try XCTUnwrap(text.range(of: "San"))
        let franciscoRange = try XCTUnwrap(text.range(of: "Francisco"))

        let detections = RampartModelDetectionBuilder.detections(
            from: [
                TokenPrediction(token: "nick", range: nickRange, label: "B-GIVEN_NAME", score: 0.92),
                TokenPrediction(token: "lives", range: text.range(of: "lives"), label: "O", score: 0.98),
                TokenPrediction(token: "san", range: sanRange, label: "B-CITY", score: 0.88),
                TokenPrediction(token: "francisco", range: franciscoRange, label: "I-CITY", score: 0.86)
            ],
            in: text,
            minimumScore: 0.4
        )

        XCTAssertEqual(detections.map(\.label), ["GIVEN_NAME", "CITY"])
        XCTAssertEqual(detections.map(\.text), ["Nick", "San Francisco"])
        XCTAssertEqual(detections.map(\.source), [.model, .model])
    }

    func testModelDetectionBuilderRepairsHyphenatedParticlesAndSentenceBoundaries() throws {
        let hyphenated = "My name is Thanh-Nghiem Quoc-Bao."
        let thanhRange = try XCTUnwrap(hyphenated.range(of: "Thanh"))
        let nghiemRange = try XCTUnwrap(hyphenated.range(of: "Nghiem"))
        let quocRange = try XCTUnwrap(hyphenated.range(of: "Quoc"))
        let baoRange = try XCTUnwrap(hyphenated.range(of: "Bao"))

        let hyphenatedDetections = RampartModelDetectionBuilder.detections(
            from: [
                TokenPrediction(token: "thanh", range: thanhRange, label: "B-GIVEN_NAME", score: 0.55),
                TokenPrediction(token: "nghiem", range: nghiemRange, label: "B-GIVEN_NAME", score: 0.22),
                TokenPrediction(token: "quoc", range: quocRange, label: "B-GIVEN_NAME", score: 0.45),
                TokenPrediction(token: "bao", range: baoRange, label: "B-GIVEN_NAME", score: 0.21)
            ],
            in: hyphenated,
            minimumScore: 0.4
        )
        XCTAssertEqual(hyphenatedDetections.map(\.text), ["Thanh-Nghiem Quoc-Bao"])

        let particle = "Applicant De La Croix applied."
        let croixRange = try XCTUnwrap(particle.range(of: "Croix"))
        let particleDetections = RampartModelDetectionBuilder.detections(
            from: [
                TokenPrediction(token: "croix", range: croixRange, label: "B-GIVEN_NAME", score: 0.61)
            ],
            in: particle,
            minimumScore: 0.4
        )
        XCTAssertEqual(particleDetections.map(\.text), ["De La Croix"])

        let boundary = "We emailed Maria. Bob replied later."
        let mariaRange = try XCTUnwrap(boundary.range(of: "Maria"))
        let bobRange = try XCTUnwrap(boundary.range(of: "Bob"))
        let boundaryDetections = RampartModelDetectionBuilder.detections(
            from: [
                TokenPrediction(token: "maria", range: mariaRange, label: "B-GIVEN_NAME", score: 0.9),
                TokenPrediction(token: "bob", range: bobRange, label: "B-GIVEN_NAME", score: 0.9)
            ],
            in: boundary,
            minimumScore: 0.4
        )
        XCTAssertEqual(boundaryDetections.map(\.text), ["Maria", "Bob"])
    }

    func testDefaultPolicyKeepsCityStateZipAndPrefersDeterministicOverlap() throws {
        let text = "Nick lives in San Francisco, CA 94105. SSN 111-11-1111."
        let nickRange = try XCTUnwrap(text.range(of: "Nick"))
        let cityRange = try XCTUnwrap(text.range(of: "San Francisco"))
        let stateRange = try XCTUnwrap(text.range(of: "CA"))
        let zipRange = try XCTUnwrap(text.range(of: "94105"))
        let ssnRange = try XCTUnwrap(text.range(of: "111-11-1111"))

        let detections = [
            PIIDetection(label: "GIVEN_NAME", range: nickRange, text: "Nick", source: .model, score: 0.9),
            PIIDetection(label: "CITY", range: cityRange, text: "San Francisco", source: .model, score: 0.9),
            PIIDetection(label: "STATE", range: stateRange, text: "CA", source: .model, score: 0.9),
            PIIDetection(label: "ZIP_CODE", range: zipRange, text: "94105", source: .model, score: 0.9),
            PIIDetection(label: "PHONE", range: ssnRange, text: "111-11-1111", source: .model, score: 0.9),
            PIIDetection(label: "SSN", range: ssnRange, text: "111-11-1111", source: .deterministic, score: 1)
        ]

        let redactable = RampartDetectionPolicy.redactableDetections(from: detections, in: text)

        XCTAssertEqual(redactable.map(\.label), ["SSN", "GIVEN_NAME"])
        XCTAssertEqual(redactable.map(\.source), [.deterministic, .model])
        XCTAssertEqual(
            RampartDetectionPolicy.redactedText(for: text, detections: redactable),
            "[GIVEN_NAME] lives in San Francisco, CA 94105. SSN [SSN]."
        )
    }

    func testClassifierIncludesDeterministicDetections() throws {
        let classifier = try RampartCoreMLClassifier(artifacts: requireArtifacts())
        let text = "my name is nick and my ssn is 111-11-1111"
        let result = try classifier.classify(text)

        XCTAssertEqual(result.modelInput, "my name is nick and my ssn is [SSN]")
        XCTAssertEqual(result.deterministicDetections.map(\.label), ["SSN"])
        XCTAssertEqual(result.deterministicDetections.first?.text, "111-11-1111")
        XCTAssertTrue(result.predictions.contains { $0.label == "B-GIVEN_NAME" })
    }

    func testGuardProtectsModelAndDeterministicDetections() throws {
        let rampartGuard = try RampartGuard(artifacts: requireArtifacts())
        let text = """
        Nick Arner lives at 123 Market Street, San Francisco, CA 94105. \
        Email nick@example.com or call 415-555-0198. His SSN is 111-11-1111.
        """
        let result = try rampartGuard.protect(text)

        XCTAssertEqual(rampartGuard.reveal(result.protectedText), text)
        XCTAssertTrue(result.protectedText.contains("[GIVEN_NAME_1]"))
        XCTAssertTrue(result.protectedText.contains("[SURNAME_1]"))
        XCTAssertTrue(result.protectedText.contains("[BUILDING_NUMBER_1]"))
        XCTAssertTrue(result.protectedText.contains("[STREET_NAME_1]"))
        XCTAssertTrue(result.protectedText.contains("San Francisco, CA 94105"))
        XCTAssertTrue(result.protectedText.contains("[EMAIL_1]"))
        XCTAssertTrue(result.protectedText.contains("[PHONE_1]"))
        XCTAssertTrue(result.protectedText.contains("[SSN_1]"))
        XCTAssertEqual(
            result.detections.map(\.label),
            ["GIVEN_NAME", "SURNAME", "BUILDING_NUMBER", "STREET_NAME", "EMAIL", "PHONE", "SSN"]
        )
    }
}

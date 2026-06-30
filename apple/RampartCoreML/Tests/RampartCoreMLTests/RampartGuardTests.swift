import XCTest
@testable import RampartCoreML

final class RampartGuardTests: XCTestCase {
    func testPlaceholderStoreAssignsStablePlaceholders() {
        let store = RampartPlaceholderStore()

        XCTAssertEqual(store.placeholder(for: "GIVEN_NAME", value: "John"), "[GIVEN_NAME_1]")
        XCTAssertEqual(store.placeholder(for: "GIVEN_NAME", value: "john"), "[GIVEN_NAME_1]")
        XCTAssertEqual(store.placeholder(for: "GIVEN_NAME", value: "Jane"), "[GIVEN_NAME_2]")
        XCTAssertEqual(store.placeholder(for: "SSN", value: "888-12-3456"), "[SSN_1]")
    }

    func testPlaceholderStoreKeepsConfiguredLabels() throws {
        let store = RampartPlaceholderStore(
            keepLabels: ["CITY"]
        )
        let text = "Austin John"
        let cityRange = try XCTUnwrap(text.range(of: "Austin"))
        let nameRange = try XCTUnwrap(text.range(of: "John"))

        let result = store.protect(
            text,
            detections: [
                PIIDetection(label: "CITY", range: cityRange, text: "Austin", source: .model, score: 1),
                PIIDetection(label: "GIVEN_NAME", range: nameRange, text: "John", source: .model, score: 1)
            ]
        )

        XCTAssertEqual(result.protectedText, "Austin [GIVEN_NAME_1]")
        XCTAssertEqual(result.placeholders, ["[GIVEN_NAME_1]"])
        XCTAssertEqual(store.reveal("Hello [GIVEN_NAME_1]."), "Hello John.")
    }

    func testGuardProtectsAndRevealsDeterministicDetections() throws {
        let rampartGuard = RampartGuard()
        let protected = try rampartGuard.protect("my ssn is 888-12-3456")

        XCTAssertEqual(protected.protectedText, "my ssn is [SSN_1]")
        XCTAssertEqual(protected.placeholders, ["[SSN_1]"])
        XCTAssertEqual(protected.detections.map(\.label), ["SSN"])
        XCTAssertEqual(rampartGuard.reveal("Your record [SSN_1] is updated."), "Your record 888-12-3456 is updated.")

        let reply = try rampartGuard.protect("Email alex@example.com before logging.")
        XCTAssertEqual(reply.protectedText, "Email [EMAIL_1] before logging.")
        XCTAssertEqual(rampartGuard.reveal(reply.protectedText), "Email alex@example.com before logging.")
    }
}

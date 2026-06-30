# Rampart Core ML

Core ML conversion, Swift package, CLI, and iOS example for Rampart's local PII
token-classification model and conversation guard.

License: [CC BY 4.0](../../LICENSE). See [NOTICE.md](NOTICE.md) for model
attribution and a summary of the Core ML conversion.

## Demo

[Watch the iOS example demo.](./docs/assets/rampart-ios-demo.mp4)

## Install

Add this repository to an app or package with Swift Package Manager:

```swift
.package(url: "https://github.com/nationaldesignstudio/rampart.git", branch: "master")
```

After a GitHub release containing the Apple package is published, prefer the
corresponding version tag instead of `branch: "master"`. Then add the
`RampartCoreML` product to your target and import it:

```swift
import RampartCoreML
```

The root `Package.swift` exposes the Apple runtime from this repository. The
nested `apple/RampartCoreML/Package.swift` exists so the Apple package can also
be developed and tested on its own.


## Model Artifacts

The Core ML package is published as a GitHub Release asset instead of committed
to Git. Swift package users can let the package download and cache it on first
use:

```swift
let rampart = try await RampartGuard.downloaded()
```

By default, the package downloads the Core ML artifact from this repository's
GitHub Release for the package version:

```text
https://github.com/nationaldesignstudio/rampart/releases/download/0.1.3/rampart-coreml-artifacts.zip
```

and caches the extracted files in Application Support:

```text
~/Library/Application Support/RampartCoreML/0.1.3/
```

The release asset is produced by this repository's release workflow. Until a
release containing `rampart-coreml-artifacts.zip` is published, run the
converter locally or pass an explicit same-repo artifact URL to
`downloaded(from:)`.

For repo-local workflows, download the same Core ML package, vocabulary, and
config files before running the model-backed tests:

```sh
scripts/download_model.sh
```

This writes:

```text
artifacts/RampartTokenClassifier.mlpackage
artifacts/rampart-hf/vocab.txt
artifacts/rampart-hf/config.json
```

These files are ignored by Git.

To reproduce the Core ML package from the upstream ONNX model instead, run the
converter:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-convert.txt
python3 scripts/convert_rampart_to_coreml.py
```

The converter downloads the Rampart ONNX/tokenizer/config files into
`artifacts/rampart-hf/` and writes the Core ML package to
`artifacts/RampartTokenClassifier.mlpackage`.

## Swift Package

The `RampartCoreML` package provides `RampartGuard`, a native Swift
conversation guard for local `protect`/`reveal` workflows. It wraps a
[WordPiece tokenizer](https://huggingface.co/docs/transformers/en/tokenizer_summary#wordpiece),
the BERT-style subword tokenizer Rampart expects, a Core ML inference wrapper,
and a deterministic recognizer for Rampart's structured classes: `SSN`,
`CREDIT_CARD`, `EMAIL`, `URL`, and `IP_ADDRESS`.

`RampartGuard` follows Rampart's conversation flow: it detects PII, applies
the default keep-set for `CITY`, `STATE`, and `ZIP_CODE`, replaces redactable
PII with stable typed placeholders, and reveals those placeholders in replies.

The lower-level `RampartCoreMLClassifier` remains available for callers that
need raw token predictions or merged detection spans.

```swift
import RampartCoreML

let rampart = try await RampartGuard.downloaded()
let protected = try rampart.protect("Alex Rivera lives at 221B Baker Street.")

// Send protected.protectedText to the model:
// "[GIVEN_NAME_1] [SURNAME_1] lives at [BUILDING_NUMBER_1] [STREET_NAME_1]."

let reply = rampart.reveal("Thanks, [GIVEN_NAME_1].")
let protectedReply = try rampart.protect("Email alex@example.com before logging.")
```

To use a custom cache location or an explicit artifact URL from this repository:

```swift
let rampart = try await RampartGuard.downloaded(
    to: cacheURL,
    from: artifactURL
)
```

From the repository root, run the Swift package checks with:

```sh
swift test
```

Tests that need local model artifacts skip until
`scripts/download_model.sh` or `scripts/convert_rampart_to_coreml.py` has been
run.

## CLI

Run ad hoc protection with:

```sh
swift run RampartCLI -- "my name is nick and my ssn is 111-11-1111"
swift run RampartCLI -- --all "my name is nick and my ssn is 111-11-1111"
```

The CLI downloads the model artifacts on first run when the default local
artifact paths are missing. It prints deterministic matches, merged
default-policy detections, placeholdered text, and model token predictions.

From inside `apple/RampartCoreML`, use the nested package directly:

```sh
cd apple/RampartCoreML
swift run RampartCLI -- "Nick Arner lives at 123 Market Street."
```

## iOS Example

An example SwiftUI app lives at:

```sh
Examples/RampartExampleiOS/RampartExampleiOS.xcodeproj
```

Open the project and run the app. The example downloads and caches the model on
first launch, then uses `RampartGuard` to show a text input, placeholdered
safe-text preview, and structured detections.

The example app does not bundle the model. This keeps the repository and app
target small while exercising the same download/cache path used by package
consumers.

## Validation

After running the converter, compare ONNX and Core ML predictions with:

```sh
python3 scripts/validate_coreml_parity.py
```

The parity harness tokenizes fixed real-text fixtures, runs both ONNX Runtime
and Core ML, and fails if active-token labels or logit drift exceed the configured
thresholds.

## License

This repository is released under the Creative Commons Attribution 4.0
International license (`CC-BY-4.0`) to match the upstream Rampart model.

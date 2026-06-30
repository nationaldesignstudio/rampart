import RampartCoreML
import SwiftUI

struct ContentView: View {
    @State private var input = SampleText.fullPII.text
    @State private var protectionResult: RampartProtectionResult?
    @State private var status = "Loading model..."
    @State private var rampartGuard: RampartGuard?
    @State private var isLoading = false
    @State private var isRunning = false

    var body: some View {
        NavigationStack {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        inputSection
                        actionRow(scrollProxy: proxy)

                        if let protectionResult {
                            GuardResultView(input: input, result: protectionResult)
                        } else {
                            StatusView(status: status, isLoading: isLoading)
                        }
                    }
                    .id(ViewAnchor.top)
                    .padding()
                }
            }
            .navigationTitle("Rampart")
        }
        .task {
            await loadGuard()
        }
    }

    private var inputSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Input", systemImage: "text.alignleft")
                    .font(.headline)

                Spacer()

                Menu {
                    ForEach(SampleText.allCases) { sample in
                        Button(sample.title) {
                            input = sample.text
                            protectionResult = nil
                            status = "Sample loaded."
                        }
                    }
                } label: {
                    Label("Samples", systemImage: "list.bullet.rectangle")
                }
            }

            TextField("Text to protect", text: $input, axis: .vertical)
                .lineLimit(5...10)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
                .textFieldStyle(.plain)
                .padding(12)
                .background(.background, in: RoundedRectangle(cornerRadius: 8))
                .overlay {
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(.quaternary, lineWidth: 1)
                }
        }
    }

    private func actionRow(scrollProxy: ScrollViewProxy) -> some View {
        HStack(spacing: 10) {
            Button {
                protect()
            } label: {
                HStack(spacing: 8) {
                    if isLoading || isRunning {
                        ProgressView()
                    } else {
                        Image(systemName: "text.viewfinder")
                    }
                    Text(isRunning ? "Protecting" : "Protect Text")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(rampartGuard == nil || isLoading || isRunning || input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

            Button {
                reset(scrollProxy: scrollProxy)
            } label: {
                Label("Reset", systemImage: "arrow.counterclockwise")
            }
            .buttonStyle(.bordered)
            .disabled(isLoading || isRunning)
        }
    }

    @MainActor
    private func loadGuard() async {
        guard rampartGuard == nil, !isLoading else { return }
        isLoading = true
        defer { isLoading = false }

        do {
            status = "Downloading model..."
            let artifacts = try await RampartModelArtifacts.downloadIfNeeded()
            status = "Loading model..."
            rampartGuard = try RampartGuard(artifacts: artifacts)
            status = "Model loaded. Ready."
        } catch {
            status = "Model load failed: \(error.localizedDescription)"
        }
    }

    private func protect() {
        guard let rampartGuard else {
            status = "Model is not loaded."
            return
        }

        isRunning = true
        defer { isRunning = false }

        do {
            protectionResult = try rampartGuard.protect(input)
        } catch {
            protectionResult = nil
            status = "Protection failed: \(error.localizedDescription)"
        }
    }

    private func reset(scrollProxy: ScrollViewProxy) {
        input = SampleText.fullPII.text
        protectionResult = nil
        status = rampartGuard == nil ? "Loading model..." : "Model loaded. Ready."

        withAnimation(.snappy) {
            scrollProxy.scrollTo(ViewAnchor.top, anchor: .top)
        }
    }

}

private enum ViewAnchor {
    static let top = "top"
}

private struct GuardResultView: View {
    let input: String
    let result: RampartProtectionResult

    private var detections: [DisplayDetection] {
        DisplayDetection.make(from: result, input: input)
    }

    private var protectedText: String {
        result.protectedText
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 10) {
                Label("Protected Text", systemImage: "eye.slash")
                    .font(.headline)

                Text(protectedText)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(12)
                    .background(.blue.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
                    .overlay {
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(.blue.opacity(0.2), lineWidth: 1)
                    }
            }

            VStack(alignment: .leading, spacing: 10) {
                Label("Detected PII", systemImage: "tag")
                    .font(.headline)

                if detections.isEmpty {
                    Text("No PII detected.")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.vertical, 8)
                } else {
                    LazyVStack(spacing: 10) {
                        ForEach(detections) { detection in
                            DetectionRow(detection: detection)
                        }
                    }
                }
            }
        }
    }
}

private struct StatusView: View {
    let status: String
    let isLoading: Bool

    var body: some View {
        HStack(spacing: 10) {
            if isLoading {
                ProgressView()
            } else {
                Image(systemName: "checkmark.circle")
                    .foregroundStyle(.green)
            }

            Text(status)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
    }
}

private struct DetectionRow: View {
    let detection: DisplayDetection

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(detection.label)
                    .font(.headline)

                Spacer()

                Text(detection.source.title)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(detection.source.foregroundStyle)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(detection.source.backgroundStyle, in: Capsule())
            }

            Text(detection.text)
                .font(.body)
                .textSelection(.enabled)

            HStack(spacing: 10) {
                Text(detection.offsetText)

                if let score = detection.score {
                    Text("score \(score.formatted(.number.precision(.fractionLength(3))))")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background, in: RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(.quaternary, lineWidth: 1)
        }
    }
}

private struct DisplayDetection: Identifiable {
    let id: String
    let label: String
    let text: String
    let range: Range<String.Index>
    let startOffset: Int
    let endOffset: Int
    let source: DetectionSource
    let score: Float?

    var offsetText: String {
        "[\(startOffset), \(endOffset))"
    }

    init(
        label: String,
        range: Range<String.Index>,
        source: DetectionSource,
        score: Float?,
        in input: String
    ) {
        self.label = label
        self.text = String(input[range])
        self.range = range
        self.startOffset = input.distance(from: input.startIndex, to: range.lowerBound)
        self.endOffset = input.distance(from: input.startIndex, to: range.upperBound)
        self.source = source
        self.score = score
        self.id = "\(source.title)-\(label)-\(startOffset)-\(endOffset)"
    }

    static func make(from result: RampartProtectionResult, input: String) -> [DisplayDetection] {
        result.detections.map { detection in
            DisplayDetection(
                label: detection.label,
                range: detection.range,
                source: DetectionSource(detection.source),
                score: detection.score,
                in: input
            )
        }
        .sorted { $0.startOffset < $1.startOffset }
    }
}

private enum DetectionSource {
    case model
    case rule

    init(_ source: PIIDetectionSource) {
        switch source {
        case .model:
            self = .model
        case .deterministic:
            self = .rule
        }
    }

    var title: String {
        switch self {
        case .model:
            "Model"
        case .rule:
            "Rule"
        }
    }

    var foregroundStyle: Color {
        switch self {
        case .model:
            .blue
        case .rule:
            .purple
        }
    }

    var backgroundStyle: Color {
        foregroundStyle.opacity(0.12)
    }
}

private enum SampleText: String, CaseIterable, Identifiable {
    case fullPII
    case contact
    case address
    case ssn

    var id: String { rawValue }

    var title: String {
        switch self {
        case .fullPII:
            "Name, address, email, phone, SSN"
        case .contact:
            "Email and phone"
        case .address:
            "Street address"
        case .ssn:
            "SSN"
        }
    }

    var text: String {
        switch self {
        case .fullPII:
            """
            Nick Arner lives at 123 Market Street, San Francisco, CA 94105.
            Email nick@example.com or call 415-555-0198.
            His SSN is 111-11-1111.
            """
        case .contact:
            "Reach Maya Patel at maya.patel@example.com or 212-555-0174."
        case .address:
            "Send the package to 88 King Street, Seattle, WA 98104."
        case .ssn:
            "My name is Nick and my SSN is 111-11-1111."
        }
    }
}

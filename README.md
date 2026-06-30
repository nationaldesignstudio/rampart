# Rampart

**Rampart** is a local-first system for removing personally identifiable information
from user-typed text before it leaves the browser. It combines a 14.7 MB ONNX
token-classification model with a deterministic recognizer layer; together they form
a defense-in-depth pipeline released as a complete, reproducible artifact.

This repository ships the runtime as [`@nationaldesignstudio/rampart`](https://www.npmjs.com/package/@nationaldesignstudio/rampart).
Model weights load from Hugging Face by default; a copy also lives in `model/` for
local serving and training.

```txt
"My name is Alex Rivera and my SSN is 472-81-0094."
→ "My name is [GIVEN_NAME_1] [SURNAME_1] and my SSN is [SSN_1]."
```

The model provider sees placeholders. The user sees restored values on the client.
The session table never leaves the device.

Rampart is **harm reduction**, not perfect protection. Personal data the client
fails to redact is the upper bound on what can leak through a model provider, a
logging pipeline, or a future infrastructure compromise. No detector at this size
catches everything; we document failure modes openly and ship regression tests so
future training runs surface drops immediately.

See [WHITEPAPER.md](./WHITEPAPER.md) for the full technical writeup — methodology,
candidate sweep, calibration, schema reconciliation, and reproducibility.

## Supported scope

This release supports **seven Latin-script languages**: English, Spanish, French,
German, Italian, Portuguese, and Dutch. Every headline number below is measured on
these languages.

Text and names in **non-Latin scripts** (e.g. Chinese, Japanese, Korean, Arabic,
Cyrillic, Devanagari) are **out of scope in this release**: recall drops sharply and
the system should not be relied on for them. See [Limitations](#limitations).

## Headline results

On a 30,000-row held-out test set spanning all **seven supported languages** from
the OpenPII 1.5M dataset, the **full system** (model + deterministic layer +
policy) achieves:

| Metric | Value |
| --- | ---: |
| Private-term recall (7 languages) | **98.42%** (Wilson 95% CI [98.35, 98.49]) |
| Public-term retention | 91.7% term-presence (>99% policy-aware\*) |
| Latency p50 (Node ONNX) | **6.6 ms**† |
| Shipped artifact size | **14.7 MB** Q4 ONNX (≈15.0 MB with tokenizer) |

\* OpenPII marks street-line components as public; Rampart redacts the precise
street line (`BUILDING_NUMBER` + `STREET_NAME`) and the secondary-address line
(`SECONDARY_ADDRESS`) while keeping city, state, and ZIP.
See "Schema reconciliation" in the whitepaper.

† Latency is hardware-dependent; the committed `eval/bench/runs/sample-900` proof run
records ≈14 ms p50 on CI hardware. Over the held-out slice the browser pipeline runs at
**3.9 ms p50** on WebGPU (Apple Metal) and 12.6 ms on WASM via `bun run bench:webgpu`.

These numbers come from the shipped Q4 pipeline scored end-to-end by the committed
`eval/bench` harness on a pinned held-out slice. The harness was corrected since the
previous revision: city/state/ZIP are now scored as kept (matching the runtime policy)
rather than counted as leaks, which is why public retention rose to ~90% while the
headline recall is reported against the larger, harder seven-language slice.

Per supported language, on the same 30,000-row test set:

| Language | Private recall | Public retention |
| --- | ---: | ---: |
| English (en) | 98.85% | 90.5% |
| Spanish (es) | 98.84% | 91.6% |
| French (fr) | 98.41% | 92.8% |
| German (de) | 97.94% | 91.7% |
| Italian (it) | 97.83% | 94.1% |
| Portuguese (pt) | 97.73% | 92.5% |
| Dutch (nl) | 97.21% | 91.9% |

On the English+Spanish slice the full system scores **98.85%** recall — the slice used
for the model-selection sweep in the whitepaper.

## Design goals

1. **Local-first privacy.** Remove personal information before it reaches application
   infrastructure. Data the server never receives cannot be leaked downstream.
2. **Browser-deployable.** Under 15 MB on the wire — small enough for a low-end phone
   over a slow connection.
3. **Recall-biased.** A miss leaks data; over-redaction is the lesser failure mode.
4. **Domain-aware retention.** The keep-set is policy-driven so assistants retain
   coarse geography — city, state, ZIP — while the precise street line is redacted,
   all without retraining.

## Architecture

Two cooperating layers run in parallel and merge their outputs. Both run entirely in
the browser.

### Deterministic recognizer layer

Regular expressions paired with checksum and structural validators. It owns five
classes end-to-end:

- **Credit cards** — Luhn-checksummed over the digit projection, so every
  separator form collapses to one rule and a 16-digit number that fails Luhn is
  not redacted as a card.
- **SSNs** — structural rules reject reserved areas (`000`, `666`, `9xx`) and
  ZIP+4-style false positives.
- **Email**, **URLs**, and **IP addresses** (IPv4, IPv6, and MAC) — pattern-backed,
  where the structure lives in the punctuation; near-100% recall, far above the
  model alone (model-only URL recall is ~5%).

This layer is synchronous and runs before the model loads; its spans are masked to
sentinels so the model never re-derives them. Names, phone numbers, account and
routing numbers, government identifiers, passports, licenses, and street-address
components carry no checksum, so they are left to the model rather than guessed at
with a regex.

### Token-classification model

A MiniLM-L6-H384 encoder fine-tuned on a 35-label BIO head (17 entity types) covers
contextual PII the regex layer can't checksum — split names (`GIVEN_NAME`,
`SURNAME`), phone numbers, account/routing/tax numbers, government IDs, passports,
licenses, and free-form address components — across seven Latin-script languages
(en, es, fr, de, it, pt, nl). Vocabulary is trimmed to 19,730 WordPieces;
single-character pieces are retained for rare-name fallback.

Span repair (adjacent merge, bridge-and-merge, capitalized-particle rescue) lifts
span-F1 to 0.53 strict (IoU=1.0) and 0.66 relaxed (IoU≥0.5) on the headline test
set — well above the fragmented spans HuggingFace's default aggregation produces
for subword-split names.

### Policy and session table

**Default-deny policy:** every detected label is redacted unless explicitly kept.
The default keep-set is `{CITY, STATE, ZIP_CODE}` — coarse geography an assistant
can reason about — while the precise street line (`BUILDING_NUMBER` + `STREET_NAME`)
and the secondary-address line (`SECONDARY_ADDRESS`) are always redacted.

**Session table:** maps each raw value to a stable placeholder (`Maria Garcia → [GIVEN_NAME_1] [SURNAME_1]`).
Placeholders are restored locally in assistant responses. The table is never transmitted.

The npm package exposes a single entry point — `createGuard()` returns a `ChatGuard`
that runs the full pipeline (detect → policy → placeholders) and keeps per-conversation
state for `reveal()`.

## Install

```bash
npm install @nationaldesignstudio/rampart @huggingface/transformers
```

`@huggingface/transformers` is a peer dependency — your app bundles and serves it.

## Usage

Create one `ChatGuard` per conversation. `createGuard()` loads the shipped q4 classifier
(`nationaldesignstudio/rampart`) from Hugging Face by default, caches it on-device, and
pairs it with the deterministic layer.

```ts
import { createGuard } from "@nationaldesignstudio/rampart";

const guard = await createGuard();

// Scrub the user message before it reaches your LLM or server.
const safe = await guard.protect(userMessage);

// Send safe.text — placeholders, not raw PII — to the model.
const reply = await llm(safe.text);

// Restore real values before showing the reply to the user.
guard.reveal(reply);
```

Streaming replies: `stream.pipeThrough(guard.revealTransform())`.

Scrub model output before logging: `await guard.protectReply(reply)`.

### Options

| Option | Default | Purpose |
| --- | --- | --- |
| `model` | `nationaldesignstudio/rampart` | Hugging Face model id or local ONNX directory |
| `device` | `"wasm"` | `"wasm"` / `"webgpu"` in browsers; `"cpu"` in Node |
| `worker` | — | Worker script URL — run NER off the main thread |
| `heuristicsOnly` | `false` | Skip the classifier; structured PII only |
| `keepLabels` | city, state, ZIP | Widen or narrow the default-deny keep-set |
| `aliases` | `{}` | Display names for tokens, e.g. `{ GIVEN_NAME: "NAME" }` |
| `ner` | — | Inject a custom detector; skips `model` |
| `minScore` | `0.4` | Drop model spans below this confidence |
| `noPrefilter` | `false` | Feed raw text to the model (no-prefilter ablation); heuristics still run |

```ts
// Local weights (e.g. self-hosted or repo `model/` dir)
const guard = await createGuard({ model: "./model", device: "cpu" });

// Heuristics only — no model load
const guard = await createGuard({ heuristicsOnly: true });

// Keep inference off the UI thread (browser)
const guard = await createGuard({
  worker: new URL("./pii-worker.ts", import.meta.url),
});
```

Custom models must be token-classification ONNX exports (q4) with a label schema
compatible with Rampart. Dtype is fixed to q4.

Set `HF_TOKEN` when pulling from a private Hugging Face repo.

### CLI

```bash
bun run redact   # interactive terminal redactor (Node, device: cpu)
```

### Apple platforms

For native iOS and macOS apps, this repository also exposes a Swift package
product, `RampartCoreML`, backed by a Core ML conversion of the released model.
The Apple sources, conversion scripts, and iOS example live in
[`apple/RampartCoreML`](./apple/RampartCoreML).

[Watch the iOS example demo.](./apple/RampartCoreML/docs/assets/rampart-ios-demo.mp4)

```swift
import RampartCoreML

let rampart = try await RampartGuard.downloaded()
let protected = try rampart.protect("My name is Alex Rivera.")
let reply = rampart.reveal("Hello [GIVEN_NAME_1].")
```

The Swift package downloads the published Core ML artifact on first use, runs
inference locally, and keeps a local placeholder table for `protect`/`reveal`.
To reproduce the Core ML package from the released ONNX model, see
[`apple/RampartCoreML/README.md`](./apple/RampartCoreML/README.md).

## Limitations

The most consequential documented gaps:

- **Non-Latin scripts are out of scope.** This release supports the seven
  Latin-script languages above only. On the fairness suite, Latin-script names —
  including diacritics — recall ~99.8%, but names in non-Latin scripts recall ~14%
  in aggregate (Russian 2%, Arabic 5%, Hindi 6%, Han Chinese 9%, Korean 15%,
  Japanese 46%). There is no checksum for names, so this gap surfaces at the system
  level. **Do not deploy this release for populations who routinely type non-Latin-script
  names without compensating controls.** Tracked by a stratified regression test in
  the eval suite; closing it is the top priority for the next training cycle.
- **Adversarial robustness.** 86.4% on a 20-case hostile-input suite. Combined
  attacks can still bypass both layers. The threat model is good-faith user entry,
  not a motivated adversary smuggling PII past their own filter.
- **Indirect identifiers.** Rare condition + ZIP-style inferential leaks are out of scope.
- **Non-text inputs.** Images, audio, and structured form fields are not supported.

See [MODEL_CARD.md](./MODEL_CARD.md) for per-class statistics and failure modes.

## Evaluation

Everything is evaluated in TypeScript against the shipped `ChatGuard` pipeline — the
same code consumers run is the code under test:

- **Unit tests** — deterministic detectors, redaction policy, streaming rehydration,
  span repair.
- **Public API end-to-end suite** — chat-style cases across structured identifiers,
  names from multiple traditions, addresses, government IDs, keep-set behavior, and
  no-PII controls.
- **Native benchmark** ([`eval/bench`](./eval/bench)) — runs the real
  `@nationaldesignstudio/rampart` pipeline over a frozen OpenPII held-out slice and
  scores recall/retention (Wilson CI), span-F1, and latency. The headline numbers
  above are regenerated by this harness.

```bash
bun test                                   # unit + public API suites
bun run eval:public                        # end-to-end chat cases

bun run bench:fetch --n 30000              # materialise the held-out slice
bun run bench                              # score the shipped pipeline
```

## Documentation

| Document | Contents |
| --- | --- |
| [WHITEPAPER.md](./WHITEPAPER.md) | Full technical writeup |
| [MODEL_CARD.md](./MODEL_CARD.md) | Model summary, training data, eval results, limitations |
| [RELEASE.md](./RELEASE.md) | Verify and publish checklist |

## Distribution

| Channel | Artifact |
| --- | --- |
| **npm** | `@nationaldesignstudio/rampart` — TypeScript runtime API |
| **GitHub** | [`nationaldesignstudio/rampart`](https://github.com/nationaldesignstudio/rampart) — source, tests, eval harness, model weights |
| **Swift / Core ML** | `RampartCoreML` — Apple-platform Core ML conversion, Swift package, CLI, and iOS example in [`apple/RampartCoreML`](./apple/RampartCoreML) |

## License

Released under [CC BY 4.0](./LICENSE) (Creative Commons Attribution 4.0 International).

Training data: OpenPII 1.5M (CC BY 4.0).

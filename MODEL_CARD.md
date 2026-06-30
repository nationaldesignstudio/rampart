---
library_name: transformers.js
pipeline_tag: token-classification
license: cc-by-4.0
language:
  - en
  - es
  - fr
  - de
  - it
  - pt
  - nl
tags:
  - pii
  - redaction
  - privacy
  - onnx
  - web
  - client-side
  - minilm
  - browser
datasets:
  - ai4privacy/pii-masking-openpii-1.5m
base_model: nreimers/MiniLM-L6-H384-uncased
metrics:
  - private-term-recall
  - public-term-retention
  - span-f1
  - ece
---

# Rampart

`rampart` is a 14.7 MB ONNX token-classification model that detects personally identifiable information (PII) in text before it leaves the user's device.
It is the on-device half of **Rampart**, a defense-in-depth client-side redaction system released by National Design Studio.
The shipped artifact runs alongside a deterministic recognizer layer that handles structured identifiers; together they form the complete system.

This card documents the released artifact only.
Alternative configurations explored during model selection (an ELECTRA-small base, the prefilter-off training variant, leaner data mixes, and smaller corpus slices) are discussed in the project whitepaper for context but are not published.

## Model summary

| Property              | Value                                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Model id              | `nationaldesignstudio/rampart`                                                                                               |
| Architecture          | [`nreimers/MiniLM-L6-H384-uncased`](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) fine-tuned with a 35-label BIO head (17 entity types) |
| Parameters            | ≈18.5M (MiniLM-L6-H384 with the trimmed 19,730-piece vocabulary; the 22.7M base figure is for the full 30,522-piece BERT vocab) |
| Quantization          | 4-bit MatMul + INT8 embedding (`onnx/model_q4.onnx`)                                                                            |
| Shipped artifact size | 14.7 MB |
| Vocabulary            | 19,730 WordPieces (trimmed from BERT-uncased's 30,522, retaining all special and single-character pieces plus frequent multi-character pieces) |
| Max sequence          | 512 tokens                                                                                                                      |
| Languages             | English, Spanish, French, German, Italian, Portuguese, Dutch (all Latin-script)                                                 |
| Runtime               | ONNX Runtime Web (WASM/WebGPU) via `transformers.js`                                                                            |
| License               | CC BY 4.0 (Creative Commons Attribution 4.0 International)                                                                       |
| Training data license | CC BY 4.0 ([`ai4privacy/pii-masking-openpii-1.5m`](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m))        |
| Released by           | National Design Studio                                                                                                          |
| Card version          | 1.0 (initial public release)                                                                                                    |

## Intended use

The model is designed for **client-side redaction of user-typed text in AI assistants and intake flows** — replacing identifying values with stable placeholders before any data is transmitted to a model provider, a server, or a logging system.

### Direct uses

- Redact user content before passing it to a hosted LLM.
- Maintain stable placeholders (`[GIVEN_NAME_1]`, `[SSN_1]`, ...) across a multi-turn conversation, with rehydration on the client.
- Preempt accidental collection of personal data in analytics, traces, and crash reports.
- Validate domain-specific redaction policies before deploying chat systems in regulated contexts.

### Out of scope

- **Stand-alone government-ID detection.**
  The model is one layer of a defense-in-depth system; it is not a replacement for the deterministic recognizer layer that ships alongside it.
  SSNs and payment cards are caught by the deterministic layer with checksum validation (structural rules and Luhn), at higher recall than the model alone.
  Phone, routing, government-ID, passport, and license numbers carry no checksum, so they are caught by the model; the deterministic layer does not attempt them.
- **Indirect / inferential identifiers.**
  A "rare disease + 5-digit ZIP" combination can re-identify someone even though neither token is in the redact-set.
  The model does not detect inferential leaks.
- **Adversarial robustness as a security guarantee.**
  We publish numbers on hostile inputs and document the failure surface; the system is positioned as harm reduction for users entering their own information in good faith, not as a security boundary against motivated adversaries.
- **Non-Latin scripts.**
  This release is scoped to the seven Latin-script languages listed above.
  Korean, Han Chinese, Japanese, Arabic, Cyrillic, and Devanagari names recall ~14% in aggregate (see "Fairness and limitations" below).
  Do not deploy this release for populations who routinely type non-Latin-script names without compensating controls; monitor accordingly.

### Usage

The runtime ships as [`@nationaldesignstudio/rampart`](https://www.npmjs.com/package/@nationaldesignstudio/rampart). `createGuard()` returns a `ChatGuard` that loads this classifier and runs the full deterministic + model pipeline:

```ts
import { createGuard } from "@nationaldesignstudio/rampart";

const guard = await createGuard();
const { text } = await guard.protect("My name is Alex Rivera and my SSN is 472-81-0094.");
// → "My name is [GIVEN_NAME_1] [SURNAME_1] and my SSN is [SSN_1]."
```

## Training data

| Source                                                                                                       | Rows used                        | License    | Role                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------ | -------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| [`ai4privacy/pii-masking-openpii-1.5m`](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) | 643,756 train + 100,000 held-out | CC BY 4.0  | Realistic chat-style PII; 7 Latin-script languages (en, es, fr, de, it, pt, nl); the OpenPII schema mapped to our 35-label BIO schema (17 entity types) |
| Synthetic generator                                                                                          | 20,000 train                     | Apache-2.0 | Class reinforcement for the 17 entity types — accent-bearing names from curated first- and last-name pools and templated structured fields, generated deliberately messy (typos, all-caps, voice-dictated and pasted-from-form phrasing, multilingual mixing, contradictory/duplicated values) so the model sees realistic disordered input, not just clean OpenPII prose |

The held-out 100,000 rows are split into two non-overlapping subsets, seeded for full reproducibility:

- **10,000 rows** for recall-floor threshold tuning.
- **30,000 rows** for the headline test results below (per-language row counts in the eval table).

The remaining 60,000 held-out rows are reserved for future evaluation and are not used in this release.

### Pre-processing

All training rows pass through the same normalization the runtime applies before tokenization: lowercase, NFKD decomposition, and combining-mark stripping.
The combining-mark step folds accents — `José` becomes `jose`, `Müller` becomes `muller` — so the model sees a single canonical form regardless of how the user typed the name.
This matches what BERT's BasicTokenizer does implicitly at inference time under `do_lower_case=True`, so the train-time and runtime distributions are identical by construction.
A guard in the training pipeline fails the run if a future tokenizer change breaks this assumption.

A re-trainer who omits this normalization step will produce a model with mismatched distributions, and recall numbers will not reproduce.

The structured classes the deterministic layer owns (`SSN`, `CREDIT_CARD`, `IP_ADDRESS`) are also masked to sentinel tokens before tokenization — both at inference (`src/premask.ts`) and during dataset construction — so the model never learns to classify raw card/SSN/IP digits and the train-time and inference-time inputs match by construction.

### Vocabulary

The full BERT-uncased vocabulary contains 30,522 WordPieces.
The shipped vocabulary retains:

1. All special tokens (`[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`).
2. All single-character pieces and their `##` continuations, which preserve WordPiece's character-level fallback for rare names.
3. All multi-character pieces appearing in the training corpus above a frequency threshold.

The shipped vocabulary is **19,730 pieces**.

### Training procedure

| Hyperparameter      | Value                            |
| ------------------- | -------------------------------- |
| Base                | nreimers/MiniLM-L6-H384-uncased  |
| Epochs              | 3                                |
| Batch size          | 32                               |
| Learning rate       | 5e-5                             |
| Weight decay        | 0.01                             |
| Max sequence length | 512                              |
| Optimizer           | AdamW                            |
| Eval strategy       | per-epoch on held-out validation |
| Save strategy       | per-epoch                        |
| Hardware            | Apple M-series MPS               |
| Total wall time     | ~3.5 hours                       |

The final epoch was selected by held-out eval loss.

## Label taxonomy

The model emits 35 BIO labels (17 entity types × {B-, I-} + O); the deterministic
recognizer layer contributes three more structured classes that are masked before
the model runs. The runtime applies a default-deny policy: every detected span is
redacted unless its label is explicitly in the keep-set.

### Redacted by default

Owned by the deterministic recognizer layer (regex + validator, masked before the model):

| Label         | Description                                                       |
| ------------- | ----------------------------------------------------------------- |
| `SSN`         | Social Security Numbers (US) — structural validation              |
| `CREDIT_CARD` | Payment card numbers — Luhn-validated                             |
| `EMAIL`       | Email addresses                                                   |
| `URL`         | URLs in user content                                              |
| `IP_ADDRESS`  | IPv4 / IPv6 / MAC addresses                                       |

Emitted by the token-classification model:

| Label               | Description                                          |
| ------------------- | ---------------------------------------------------- |
| `GIVEN_NAME`        | Given / first names                                  |
| `SURNAME`           | Family / last names                                  |
| `PHONE`             | Phone numbers                                        |
| `TAX_ID`            | Tax identifiers                                      |
| `BANK_ACCOUNT`      | Bank account / IBAN numbers                          |
| `ROUTING_NUMBER`    | Bank routing numbers                                 |
| `GOVERNMENT_ID`     | Government-issued ID / case numbers                  |
| `PASSPORT`          | Passport numbers                                     |
| `DRIVERS_LICENSE`   | Driver's license numbers                             |
| `BUILDING_NUMBER`   | Street-line building number                          |
| `STREET_NAME`       | Street name                                          |
| `SECONDARY_ADDRESS` | Secondary-address line (apt / unit / suite)          |

`BUILDING_NUMBER` + `STREET_NAME` together form the precise street line; both are
redacted while city/state/ZIP are kept.

### Kept by default

| Label      | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| `CITY`     | City — coarse geography for eligibility checks               |
| `STATE`    | State / region                                               |
| `ZIP_CODE` | Postal code                                                  |

The keep-set keeps coarse geography (city/state/ZIP) while redacting the precise
street line. To change it, edit `KEEP_LABELS` in `src/types.ts` — it is a
compile-time set, not a runtime flag.

The taxonomy is deliberately **atomic**: there is no coarse `PERSON`,
`STREET_ADDRESS`, `ADDRESS`, `ORGANIZATION`, or `LOCATION` label, and no catch-all
`SECRET`. Names split into `GIVEN_NAME` / `SURNAME`, the street line into
`BUILDING_NUMBER` / `STREET_NAME`, and document identifiers into their specific
classes, so the model learns to catch PII fragments in disordered text rather than
expecting one tidy blob. Dates, ages, and income are intentionally **not** modeled
as PII (they map to `O`): a bare date is rarely identifying, and assistants need age
and income as context, so redacting them was over-redaction without a privacy gain.

## Evaluation

We score the **full system** (model + deterministic layer) because that is what consumers experience end-to-end.
Model-only numbers are reported separately for researchers who want to evaluate the encoder in isolation.

### Primary metrics

- **Private-term recall**: for every gold private value, did the redacted output contain the value? This is the privacy-headline number; misses here are leaks.
- **Public-term retention**: for every gold public value, did the redacted output preserve the value? This measures over-redaction.
- **Span F1 strict (IoU=1.0)** and **relaxed (IoU≥0.5)**: how well predicted span boundaries align with gold boundaries under one-to-one greedy matching.
- **Latency**: Node.js ONNX runtime cold / p50 / p95 / p99 over the full 30,000-row test set. Browser latency (WebGPU and WASM backends) is measured separately by `eval/bench/webgpu.ts` — see below.
- **Calibration**: 15-bin reliability ECE, per label and overall, on per-span max-class scores.

All recall and retention numbers carry Wilson 95% confidence intervals; stratified breakdowns include 1000-iteration bootstrap intervals.

### Held-out OpenPII test set — seven supported languages (30,000 rows; 131,707 private terms; 87,207 public terms)

The headline number is measured across all seven supported Latin-script languages.
English-only, Spanish-only, and the English+Spanish slice are reported as sub-slices.

| Slice                        | Private recall (Wilson 95%) | Public retention\* | Span F1 strict | Latency p50 |
| ---------------------------- | --------------------------- | ------------------ | -------------- | ----------- |
| **All seven languages**      | **98.42% [98.35, 98.49]**   | 91.69%             | 0.528          | 6.6 ms      |
| English only (11,569 rows)   | 98.85%                      | 90.5%              | —              | 6.6 ms      |
| Spanish only (3,234 rows)    | 98.84%                      | 91.6%              | —              | 6.6 ms      |
| English + Spanish            | 98.85%                      | 91.0%              | —              | 6.6 ms      |

2,082 leaks of 131,707 private terms on the seven-language test (1 in 64 terms slips past
the system, before the application's downstream defenses fire). On the English+Spanish
slice the system leaks 778 of 67,613.

These numbers are measured by the committed `eval/bench` harness running the **shipped Q4
pipeline** end-to-end over a pinned held-out slice of `pii-masking-openpii-1.5m`. The
harness was corrected relative to earlier revisions of this card: city/state/ZIP are now
scored as **kept** (matching the runtime keep-set) instead of being counted as leaks, so
public retention reflects policy-aware behavior directly. Recall is reported against the
full, harder seven-language slice. Span-F1 strict (exact byte+label match) is a secondary
metric; term-presence recall is the privacy headline.

The 6.6 ms p50 above is the Node ONNX (CPU) figure over the 30k held-out set. Run over a
held-out OpenPII slice in the browser, the same shipped pipeline measures **3.9 ms p50**
on WebGPU (Apple Metal, p95 9.3 ms) and 12.6 ms on WASM (p95 35.5 ms), via
`eval/bench/webgpu.ts` — so the WebGPU form factor is faster than Node CPU on the same
class of inputs, and WASM is the floor when no GPU is available.

\* See "Schema reconciliation" below — the Rampart policy redacts the precise street line
(`BUILDING_NUMBER` + `STREET_NAME`) and the secondary-address line while keeping city/state/ZIP, which the harness now honors.

### Per-language slices (OpenPII Latin test, 30k rows across 7 languages)

| Language          | Rows   | Private recall | Public retention | Leaks / total |
| ----------------- | ------ | -------------- | ---------------- | ------------- |
| English (`en`)    | 11,569 | 98.85%         | 90.5%            | 618 / 53,877  |
| Spanish (`es`)    | 3,234  | 98.84%         | 91.6%            | 160 / 13,736  |
| French (`fr`)     | 4,708  | 98.41%         | 92.8%            | 317 / 19,906  |
| German (`de`)     | 4,260  | 97.94%         | 91.7%            | 357 / 17,347  |
| Italian (`it`)    | 3,218  | 97.83%         | 94.1%            | 301 / 13,855  |
| Portuguese (`pt`) | 1,485  | 97.73%         | 92.5%            | 147 / 6,467   |
| Dutch (`nl`)      | 1,526  | 97.21%         | 91.9%            | 182 / 6,519   |

All seven languages land in the 97-99% band; Dutch is the lowest at 97.21% and is flagged
for attention in subsequent training cycles. (The recall band moved down ~1pp versus the
previous card because the harness now scores the corrected, harder slice — see the note
above; the same model scores higher on the older, easier slice.)

### Hand-curated suites

| Suite                                                                                      | Cases | Private recall (Wilson 95%) | Public retention |
| ------------------------------------------------------------------------------------------ | ----- | --------------------------- | ---------------- |
| Domain intake                                                                              | 20    | 96.97% [84.68, 99.46]       | 93.2%            |
| Adversarial (homoglyph / zero-width / leet / splits / NFC-NFD / casing / prompt-injection) | 20    | 86.36% [66.66, 95.25]       | 83.3%            |
| Fairness (Faker × 15 naming traditions × 5 templates)                                      | 1,875 | 65.44% [63.26, 67.56]       | 90.0%            |

The adversarial and domain-intake suites are 20 cases each; Wilson CIs are wide.
The 1,875-case fairness suite has tight CIs and is the most statistically grounded slice we report.

### Schema reconciliation

The 91.69% retention number in the headline table is term-presence scoring that already credits city/state/ZIP as kept, matching the runtime keep-set.
We analyzed the 7,244 remaining "over-redacted" public terms in the 30,000-row eval:

- **The vast majority** are policy-driven redactions of street-line components (street name, building number, secondary address line).
  AI4Privacy OpenPII marks `STREET`, `BUILDINGNUM`, and `SECADDRESS` as `O` (public); the Rampart policy redacts the precise street line (`BUILDING_NUMBER` + `STREET_NAME`) and `SECONDARY_ADDRESS` while keeping `CITY`, `STATE`, and `ZIP`.
  These are not detector errors; they are the policy firing as designed.
- **A smaller share** are span-edge artifacts.
  The runtime's particle-rescue step grows name spans (`GIVEN_NAME` / `SURNAME`) to swallow capitalized particles ("de la", "von", "Mc").
  When an adjacent public token is itself capitalized, that token can be absorbed into the redacted span.
- **A very small fraction** are digit fragments inside longer correctly-redacted spans (e.g. "376" found inside a redacted 16-digit credit card).

We publish the 91.69% term-presence number for like-for-like comparison against public PII benchmarks running the same scoring rules.
For product reasoning, the policy-aware retention exceeds 99%.

## Calibration

The runtime applies a single recall-biased confidence floor (`minScore` = 0.4) uniformly
across the model's labels, chosen against the 10,000-row OpenPII Latin calibration split
(disjoint from test) so misses — which leak data — are traded against the cheaper failure
of over-redaction. There is no per-label threshold table in the shipped runtime; the
deterministic recognizer layer, not a tuned model threshold, is the system of record for
the structured classes the model alone is weakest on:

- **SSN** — structural validation (reserved-area rules).
- **CREDIT_CARD** — Luhn checksum over the digit projection.
- **EMAIL / URL / IP_ADDRESS** — pattern-anchored regex at near-100% recall.

Phone, routing, government-ID, passport, and license numbers carry no checksum and are
left to the model under the same recall-biased floor.

ECE on the full 30,000-row test set is **0.291** (overall, all labels); the model alone (no deterministic layer) is **0.018**.
The system-level ECE is higher because the deterministic layer always emits score 1.0 on its detections, making the score distribution bimodal — that is a score-distribution artifact of the union, not a calibration regression of the underlying model.

## Fairness and limitations

We document failures because consumers need this to deploy the redactor responsibly.
None of these are surprises; we measured each.

### Fairness across naming traditions (1,875 Faker-generated cases)

Cases are stratified by **naming tradition** (15 categories) embedded in 5 chat templates.
Same surrounding context across all traditions — only the name varies.

| Tradition           | Locale       | Recall | Cases |
| ------------------- | ------------ | ------ | ----- |
| Anglo               | en_US        | 99.9%  | 125   |
| Hispanic            | es_MX, es_ES | 99.9%  | 250   |
| Francophone         | fr_FR        | 99.9%  | 125   |
| Germanic            | de_DE        | 99.9%  | 125   |
| Romance (Italian)   | it_IT        | 99.9%  | 125   |
| Lusophone           | pt_BR        | 99.9%  | 125   |
| Turkic              | tr_TR        | 99.9%  | 125   |
| Vietnamese          | vi_VN        | 99.2%  | 125   |
| Japanese            | ja_JP        | 45.6%  | 125   |
| Korean              | ko_KR        | 15.2%  | 125   |
| Han Chinese         | zh_CN        | 8.8%   | 125   |
| South Asian (Hindi) | hi_IN        | 5.6%   | 125   |
| Arabic              | ar_AA        | 4.8%   | 125   |
| Slavic (Russian)    | ru_RU        | 2.4%   | 125   |
| South Asian (Tamil) | ta           | 48.0%  | 125   |
| South Asian (Tamil Romanized) | ta_Latn | 99.4% | 125   |

Aggregated by script:

- **Latin-ASCII names**: ~100% recall (695 / 695)
- **Latin + diacritics**: 99.8% recall (429 / 430)
- **Non-Latin scripts**: 13.7% recall (103 / 750)

*(Note: The combined fairness-suite aggregate (n=1,875, 65.44%) is left unchanged, as the underlying case-level data for the other 15 traditions is not available in this repository to support a recomputed blended total. A maintainer with access to the original per-case results should recompute the full aggregate if Tamil is merged into the suite.)*

The deterministic recognizer layer does not catch names — there is no checksum to validate against — so this failure surfaces at the system level.
This is the most important regression we have identified, and the fairness suite is wired into the eval pipeline as a stratified regression test so any further drop will surface in subsequent training cycles.

### Government-style identifiers (model only)

Government-style identifiers (case numbers, Medicare-style identifiers, USCIS receipts,
A-numbers, passports, licenses) carry no checksum, so — unlike SSNs and payment cards —
the deterministic layer does **not** detect them. They rely entirely on the model, which
catches ~67.6% of them in a structured-ID probe.
This is a documented weak spot: there is no deterministic backstop for these classes, so
the model's recall is effectively the system's recall on them.
Consumers should not assume the deterministic layer covers government IDs the way it
covers SSNs and cards; deployments that handle these identifiers heavily should add their
own format-specific validators.

### Adversarial robustness

The system catches most homoglyph, casing, leet, NFC/NFD, and basic whitespace-split attacks.
It does not reliably catch:

- Zero-width characters injected between every digit of an SSN.
- Prompt-injection text inside the PII span (e.g. `"ignore previous instructions"`).
- Combined attacks (homoglyph plus whitespace split).

The deterministic layer's digit projection (which strips non-digit characters before checksum validation) restores most digit-bearing PII against these attacks; names remain vulnerable.
This is the right framing for the limitation, not the primary use case: Rampart is designed to protect users entering their own information in good faith from incidental disclosure to downstream services, not to defeat a motivated user actively trying to smuggle their own PII past the filter.

### WordPiece fragmentation on long names

Names like `Thanh-Nghiem Quoc-Bao` or `Chukwuemeka Okonkwo-Adeyemi` produce many subwords; the runtime performs span-merging across same-label adjacencies plus particle-rescue, which closes most of the gap.
Some five-or-more-subword names still fragment in a way that loses recall on the trailing subword.

## Reproducibility

The model weights, deterministic layer, and TypeScript evaluation harness are released under CC BY 4.0.

Evaluation runs entirely in TypeScript, against the shipped pipeline: the native
benchmark (`eval/bench`) runs the real `@nationaldesignstudio/rampart` code over a
frozen OpenPII held-out slice and writes `summary.json` / `by_language.json`, which are
committed alongside the eval output — so every number in this card traces to committed
evidence produced by the code that ships. The held-out
row `uid`s are pinned in a committed manifest; regenerate the data with
`bun run bench:fetch` and reproduce the figures with `bun run bench`.

## Citation

If you use this model in research, please cite:

```bibtex
@misc{rampart-2026,
  author = {National Design Studio},
  title  = {Rampart: Client-side PII redaction for AI assistants},
  year   = {2026},
  url    = {https://huggingface.co/nationaldesignstudio/rampart},
}
```

Please also cite the upstream training corpus:

```bibtex
@misc{ai4privacy-openpii-1.5m,
  title  = {ai4privacy/pii-masking-openpii-1.5m},
  author = {AI4Privacy},
  year   = {2025},
  url    = {https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m},
}
```

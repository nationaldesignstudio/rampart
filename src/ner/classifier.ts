/**
 * Contextual PII detection via a small token-classification model running in
 * the browser (transformers.js → ONNX Runtime Web, wasm or WebGPU backend).
 *
 * This is the residual layer: it catches what the heuristics can't — people's
 * names, organizations, and free-text identifiers — which is exactly the PII we
 * never want in our logs. The model is intentionally tiny and int8-quantized so
 * it loads once (cached in IndexedDB by the runtime) and runs on-device with no
 * server round-trip and no shared queue to saturate.
 *
 * Label mapping: the fine-tuned model emits token-classification entity groups,
 * which we map onto our {@link PiiLabel} set. CITY/STATE/ZIP_CODE are emitted
 * too so the merge step can carry them through to the keep-set.
 */

import { mergeSpans } from "../policy";
import type { PiiLabel, Span } from "../types";

/** Minimal shape of a transformers.js token-classification result row. */
interface RawEntity {
  readonly entity_group?: string;
  readonly entity?: string;
  readonly score: number;
  readonly start: number;
  readonly end: number;
  readonly word: string;
  /** Index into `[CLS] + content + [SEP]`; used for tokenizer-backed offset recovery. */
  readonly index?: number;
}

/** Counts the model tokens in a string, excluding the [CLS]/[SEP] specials. */
export type TokenCounter = (text: string) => number;

/**
 * The callable returned by a token-classification pipeline. `countTokens` is
 * attached when the classifier is backed by a real tokenizer (see
 * {@link loadNerClassifier}); {@link detectNer} uses it to size windows by the
 * model's token budget. Bare mocks may omit it, in which case detection runs the
 * whole input as a single window.
 */
export interface TokenClassifier {
  (text: string, options?: { aggregation_strategy?: "simple" | "first" | "max" }): Promise<RawEntity[]>;
  countTokens?: TokenCounter;
  /** WordPiece tokens for `text` (no specials); drives index→char offset recovery. */
  tokenize?: (text: string) => readonly string[];
}

/** Maps model entity groups to our labels. Unknown groups are dropped. */
const GROUP_TO_LABEL: Readonly<Record<string, PiiLabel>> = {
  // Split names (a household may share a surname, so they stay distinct).
  GIVEN_NAME: "GIVEN_NAME",
  GIVENNAME: "GIVEN_NAME",
  SURNAME: "SURNAME",
  LASTNAME: "SURNAME",
  // Contact / document identifiers.
  EMAIL: "EMAIL",
  PHONE: "PHONE",
  URL: "URL",
  TAX_ID: "TAX_ID",
  BANK_ACCOUNT: "BANK_ACCOUNT",
  ROUTING_NUMBER: "ROUTING_NUMBER",
  GOVERNMENT_ID: "GOVERNMENT_ID",
  PASSPORT: "PASSPORT",
  DRIVERS_LICENSE: "DRIVERS_LICENSE",
  // Address components.
  BUILDING_NUMBER: "BUILDING_NUMBER",
  STREET_NAME: "STREET_NAME",
  SECONDARY_ADDRESS: "SECONDARY_ADDRESS",
  SECADDRESS: "SECONDARY_ADDRESS",
  CITY: "CITY",
  STATE: "STATE",
  ZIP_CODE: "ZIP_CODE",
};

/** The shipped Rampart token-classifier on Hugging Face (q4 ONNX only). */
export const RAMPART_MODEL_ID = "nationaldesignstudio/rampart";

export interface NerOptions {
  /**
   * Hugging Face model id or local directory path. Must be a token-classification
   * ONNX export compatible with Rampart's label schema. Defaults to
   * {@link RAMPART_MODEL_ID}.
   */
  readonly model?: string;
  /** Backend. `"wasm"`/`"webgpu"` in browsers; `"cpu"` for Node (ORT). */
  readonly device?: "wasm" | "webgpu" | "cpu";
  /** Spans below this score are discarded. Low default → recall-biased. */
  readonly minScore?: number;
}

const DEFAULT_OPTIONS: Required<Omit<NerOptions, "model">> = {
  device: "wasm",
  minScore: 0.4,
};

/**
 * The MiniLM token classifier has a hard context window, so each NER window is
 * sized to the model's token budget — measured with the model's own tokenizer,
 * not a character proxy, so a window holds exactly as much text as actually fits
 * regardless of token density. Past this, ORT would silently truncate the
 * sequence and drop whatever followed.
 */
const MODEL_MAX_TOKENS = 512;
/** [CLS] + [SEP] the pipeline wraps every window in. */
const SPECIAL_TOKENS = 2;
/** Per-window content-token budget: the model max less specials and a safety margin. */
export const NER_TOKEN_BUDGET = MODEL_MAX_TOKENS - SPECIAL_TOKENS - 10;

/**
 * Tokens shared by consecutive NER windows. Long input slides a window of
 * {@link NER_TOKEN_BUDGET} tokens; this overlap guarantees an entity landing on a
 * window seam is still *wholly* inside a neighbouring window.
 *
 * The invariant: as long as the overlap is at least the longest entity we detect
 * (names, orgs, street lines — all a handful of tokens), no entity is ever split
 * across a boundary, so a window-edge name is never silently dropped. The generous
 * margin over the longest entity also means a seam entity reappears deep inside its
 * neighbour with ample context, which the classifier needs to label it confidently.
 */
export const NER_TOKEN_OVERLAP = 64;

/** Unicode combining marks; stripped during model-space folding (José → jose). */
const COMBINING_MARKS_RE = /\p{M}/gu;

const EXTEND_SCORE = 0.15;
const CONNECTOR_RE = /^[\s'\u2019.-]*$/;
const PERSON_LABELS: ReadonlySet<PiiLabel> = new Set(["GIVEN_NAME", "SURNAME"]);
const LEFT_PARTICLE_RE = /([\p{Lu}][\p{L}\p{M}\u2019']{0,3})([\s'\u2019.-]{1,3})$/u;
const RIGHT_PARTICLE_RE = /^([\s'\u2019.-]{1,3})([\p{Lu}][\p{L}\p{M}\u2019']{0,3})/u;

/**
 * Lazily construct the token-classification pipeline. transformers.js is a peer
 * dependency and a heavy import, so it is loaded on first use, not at module
 * load — keeping the heuristic path dependency-free.
 */
export async function loadNerClassifier(options: NerOptions = {}): Promise<TokenClassifier> {
  const { pipeline } = await import("@huggingface/transformers");
  const merged = { ...DEFAULT_OPTIONS, ...options };
  const model = merged.model ?? RAMPART_MODEL_ID;
  const classifier = await pipeline("token-classification", model, {
    dtype: "q4",
    device: merged.device,
  });
  // transformers.js's published types omit `aggregation_strategy` from
  // `TokenClassificationPipelineOptions` even though the runtime accepts it,
  // so we wrap the pipeline in a typed adapter rather than coercing the
  // union return through a double cast at the call site.
  const adapter: TokenClassifier = (text, opts) =>
    (classifier as (input: string, options?: unknown) => Promise<RawEntity[]>)(text, opts);
  // Expose the pipeline's tokenizer so detectNer can size windows by real tokens.
  // `encode` returns the content token ids (no specials), so its length is the
  // token count we budget each window against. A token-classification pipeline
  // always carries a tokenizer; guard anyway so an unexpected runtime degrades to
  // the single-window path rather than throwing mid-detection.
  const tokenizer = (classifier as unknown as {
    tokenizer?: {
      encode?: (t: string, o?: { add_special_tokens?: boolean }) => number[];
      tokenize?: (t: string) => string[];
    };
  }).tokenizer;
  if (tokenizer?.encode) {
    adapter.countTokens = (text) => tokenizer.encode!(text, { add_special_tokens: false }).length;
  }
  if (tokenizer?.tokenize) {
    adapter.tokenize = (text) => tokenizer.tokenize!(text);
  }
  return adapter;
}

/**
 * Detect contextual PII across the whole input, regardless of length.
 *
 * The model has a fixed token budget, so input longer than one window is scanned
 * as a sliding window sized to {@link NER_TOKEN_BUDGET} *tokens* (measured by the
 * classifier's own tokenizer) that overlaps its neighbour by {@link NER_TOKEN_OVERLAP}
 * tokens. Each window's spans are shifted back into whole-text coordinates; because
 * windows overlap, an entity on a seam is re-detected in both, so {@link mergeSpans}
 * collapses the duplicates into the canonical disjoint set. Input that fits one
 * window — or any classifier without a tokenizer, e.g. a bare test mock — takes a
 * single-window fast path identical to scanning the text directly.
 *
 * Sizing by tokens rather than a char cap means a window holds exactly as much
 * text as the model can attend to, and nothing past a fixed char count is silently
 * dropped: the overlap keeps any entity from being split across a seam.
 */
export async function detectNer(
  raw: string,
  classifier: TokenClassifier,
  minScore: number = DEFAULT_OPTIONS.minScore,
): Promise<Span[]> {
  const windows =
    classifier.countTokens === undefined
      ? [{ start: 0, end: raw.length }]
      : planTokenWindows(raw, classifier.countTokens, NER_TOKEN_BUDGET, NER_TOKEN_OVERLAP);

  if (windows.length <= 1) {
    return detectNerWindow(raw, classifier, minScore);
  }

  const spans: Span[] = [];
  for (const window of windows) {
    // Windows run sequentially: they share one model/session, which is not safe
    // to drive with concurrent inference calls.
    const windowSpans = await detectNerWindow(raw.slice(window.start, window.end), classifier, minScore);
    for (const span of windowSpans) {
      spans.push({ ...span, start: span.start + window.start, end: span.end + window.start });
    }
  }
  return mergeSpans(spans);
}

/** A half-open char window `[start, end)` into the raw text. */
interface CharWindow {
  readonly start: number;
  readonly end: number;
}

/**
 * Plan the sliding windows over `raw`: each holds at most `budget` tokens and
 * overlaps its predecessor by at least `overlap` tokens, together covering the
 * whole string. Windows snap to word boundaries (so no window cuts a word — and
 * therefore a token — in half); a single word longer than the budget is the only
 * case hard-split mid-word, by character, as a fallback. `countTokens` is the
 * model's tokenizer, so `budget` is the real per-window capacity.
 */
function planTokenWindows(
  raw: string,
  countTokens: TokenCounter,
  budget: number,
  overlap: number,
): CharWindow[] {
  const segments = toSegments(raw, countTokens, budget);
  if (segments.length === 0) return [];

  const windows: CharWindow[] = [];
  let i = 0;
  while (i < segments.length) {
    // Grow [i, j) while it fits the budget, always taking at least one segment
    // (toSegments guarantees each segment is within budget).
    let tokens = 0;
    let j = i;
    while (j < segments.length && (j === i || tokens + segments[j].tokens <= budget)) {
      tokens += segments[j].tokens;
      j++;
    }
    windows.push({ start: segments[i].start, end: segments[j - 1].end });
    if (j === segments.length) break;

    // Advance so the next window overlaps this one by >= overlap tokens, while
    // always making progress (start strictly after i).
    let shared = 0;
    let next = j;
    while (next > i + 1 && shared < overlap) {
      next--;
      shared += segments[next].tokens;
    }
    i = next;
  }
  return windows;
}

/** A word-aligned slice of the raw text with its model-token count. */
interface Segment {
  readonly start: number;
  readonly end: number;
  readonly tokens: number;
}

/**
 * Partition `raw` into word-aligned segments (a word plus its trailing
 * whitespace) tagged with token counts. A word that alone exceeds `budget` —
 * pathological, e.g. a long unbroken blob — is hard-split by character so every
 * returned segment fits the budget and the packer can always place it.
 */
function toSegments(raw: string, countTokens: TokenCounter, budget: number): Segment[] {
  const segments: Segment[] = [];
  for (const [start, end] of wordSpans(raw)) {
    let from = start;
    while (from < end) {
      const tokens = countTokens(raw.slice(from, end));
      if (tokens <= budget) {
        segments.push({ start: from, end, tokens });
        break;
      }
      const cut = fitCharsToBudget(raw, from, end, budget, countTokens);
      segments.push({ start: from, end: cut, tokens: countTokens(raw.slice(from, cut)) });
      from = cut;
    }
  }
  return segments;
}

/** Yield contiguous `[start, end)` spans of `raw`, each a word + trailing whitespace. */
function* wordSpans(raw: string): Generator<[number, number]> {
  const n = raw.length;
  let i = 0;
  while (i < n) {
    let j = i;
    while (j < n && !/\s/.test(raw[j])) j++; // the word
    while (j < n && /\s/.test(raw[j])) j++; // its trailing whitespace
    yield [i, j];
    i = j;
  }
}

/**
 * Largest char offset `cut` in `(from, end]` whose slice fits `budget` tokens.
 * Only reached for a single over-budget word, where token count grows with
 * length; binary search lands on the cut and the budget's safety margin absorbs
 * the slight non-monotonicity of subword tokenization at the edge.
 */
function fitCharsToBudget(
  raw: string,
  from: number,
  end: number,
  budget: number,
  countTokens: TokenCounter,
): number {
  let lo = from + 1;
  let hi = end;
  let best = from + 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (countTokens(raw.slice(from, mid)) <= budget) {
      best = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return best;
}

/**
 * Run the classifier over a single window and convert its output to {@link Span}s
 * over that window's text. Below-threshold and zero-width spans are dropped;
 * recognized kept-classes are preserved as spans so the policy layer can shield
 * them.
 *
 * Handles two pipeline output shapes:
 *  - aggregated (`entity_group` + char `start`/`end`) — when the runtime applied
 *    `simple` aggregation; offsets are used directly.
 *  - raw BIO tokens (`entity` = `B-GIVEN_NAME`/`I-GIVEN_NAME`, no char offsets) —
 *    what transformers.js emits here. When the runtime supplies a token `index`,
 *    char offsets come from a tokenizer walk; otherwise the folded `word` is
 *    located in a matching folded projection (accent-safe fallback).
 */
async function detectNerWindow(
  raw: string,
  classifier: TokenClassifier,
  minScore: number = DEFAULT_OPTIONS.minScore,
): Promise<Span[]> {
  const inferText = raw.replaceAll("-", " ");
  const entities = await classifier(inferText, { aggregation_strategy: "simple" });
  // `inferText` and `raw` are the same length (hyphen→space is 1:1), so offsets
  // recovered against this folded projection address `raw` too.
  const folded = foldForModel(inferText);
  const indexOffsets =
    classifier.tokenize === undefined ? undefined : buildTokenIndexOffsets(inferText, classifier.tokenize(inferText));
  const candidates: Span[] = [];
  for (const entity of mergeBioTokens(entities, folded, indexOffsets)) {
    const label = GROUP_TO_LABEL[entity.group.toUpperCase()];
    if (label === undefined) continue;
    if (entity.score < EXTEND_SCORE || entity.end <= entity.start) continue;
    candidates.push({
      start: entity.start,
      end: entity.end,
      label,
      score: entity.score,
      source: "ner",
      text: raw.slice(entity.start, entity.end),
    });
  }
  return repairSpans(raw, candidates, minScore);
}

interface AggregatedEntity {
  group: string;
  score: number;
  start: number;
  end: number;
}

/** Strip a BIO prefix: `B-GIVEN_NAME`/`I-GIVEN_NAME` → `GIVEN_NAME`; bare labels pass through. */
function stripBio(label: string): { prefix: "B" | "I" | null; base: string } {
  const m = /^([BI])-(.+)$/.exec(label);
  return m ? { prefix: m[1] as "B" | "I", base: m[2] } : { prefix: null, base: label };
}

/** A folded copy of the model input plus a map from each folded char to raw. */
interface FoldedProjection {
  /** Lowercased, NFKD, combining-mark-stripped copy of the input. */
  readonly text: string;
  /** `rawStart[i]` is the raw offset of the source code point of `text[i]`. */
  readonly rawStart: number[];
  /** `rawEnd[i]` is the raw offset just past that source code point. */
  readonly rawEnd: number[];
}

/**
 * Fold `raw` to the model's normalized space (lowercase + NFKD + combining-mark
 * strip — the same fold BERT's BasicTokenizer applies) while recording, per
 * folded character, the `[start, end)` raw offsets of the code point it came
 * from. This is the bridge that lets a folded token `word` be matched against
 * folded text and then projected back to exact raw offsets. Iterates by code
 * point so surrogate pairs map back to whole-character raw spans.
 */
function foldForModel(raw: string): FoldedProjection {
  let text = "";
  const rawStart: number[] = [];
  const rawEnd: number[] = [];
  let i = 0;
  for (const codePoint of raw) {
    const folded = codePoint.toLowerCase().normalize("NFKD").replace(COMBINING_MARKS_RE, "");
    for (const ch of folded) {
      text += ch;
      rawStart.push(i);
      rawEnd.push(i + codePoint.length);
    }
    i += codePoint.length;
  }
  return { text, rawStart, rawEnd };
}

/**
 * Map each pipeline token index (`[CLS]` = 0, content, `[SEP]` last) to char
 * offsets in `inferText`. Content offsets are recovered by walking the
 * tokenizer's WordPiece pieces in order — so the second `123` in a sentence
 * lands on the address token, not the first digit run from `indexOf`.
 */
function buildTokenIndexOffsets(
  inferText: string,
  contentTokens: readonly string[],
): ReadonlyArray<readonly [number, number]> {
  const offsets: [number, number][] = [[0, 0]];
  let cursor = 0;
  const lower = inferText.toLowerCase();
  for (const token of contentTokens) {
    if (token.startsWith("##")) {
      const piece = token.slice(2);
      offsets.push([cursor, cursor + piece.length]);
      cursor += piece.length;
    } else {
      const at = lower.indexOf(token, cursor);
      if (at < 0) {
        offsets.push([0, 0]);
      } else {
        offsets.push([at, at + token.length]);
        cursor = at + token.length;
      }
    }
  }
  offsets.push([0, 0]);
  return offsets;
}

function offsetFromTokenIndex(
  indexOffsets: ReadonlyArray<readonly [number, number]> | undefined,
  index: number | undefined,
): readonly [number, number] | null {
  if (indexOffsets === undefined || index === undefined) return null;
  const pair = indexOffsets[index];
  if (pair === undefined || pair[0] === pair[1]) return null;
  return pair;
}

/**
 * Normalize either output shape into raw-offset entities. Aggregated rows carry
 * offsets in the model-input (unfolded) coordinate system, so they are used
 * directly. Raw BIO tokens are merged (B starts a span, matching I extends it);
 * when the runtime supplies a token `index`, char offsets come from the
 * tokenizer walk. Otherwise each token's folded `word` is located in the folded
 * projection via a forward-advancing search and projected back to raw.
 */
function mergeBioTokens(
  entities: RawEntity[],
  folded: FoldedProjection,
  indexOffsets?: ReadonlyArray<readonly [number, number]>,
): AggregatedEntity[] {
  const out: AggregatedEntity[] = [];
  let cursor = 0;
  let current: (AggregatedEntity & { count: number }) | null = null;

  const flush = (): void => {
    if (current !== null) {
      out.push({ group: current.group, score: current.score / current.count, start: current.start, end: current.end });
      current = null;
    }
  };

  for (const entity of entities) {
    // Already-aggregated shape: offsets are in unfolded model-input coordinates.
    if (entity.entity_group !== undefined && typeof entity.start === "number" && typeof entity.end === "number") {
      flush();
      out.push({ group: entity.entity_group, score: entity.score, start: entity.start, end: entity.end });
      continue;
    }
    const rawLabel = entity.entity ?? entity.entity_group;
    if (rawLabel === undefined) continue;
    const { prefix, base } = stripBio(rawLabel);

    const indexed = offsetFromTokenIndex(indexOffsets, entity.index);
    let start: number;
    let end: number;
    if (indexed !== null) {
      [start, end] = indexed;
    } else {
      const word = (entity.word ?? "").replace(/^##/, "").toLowerCase();
      if (!word) continue;
      const at = folded.text.indexOf(word, cursor);
      if (at < 0) continue;
      start = folded.rawStart[at];
      end = folded.rawEnd[at + word.length - 1];
      cursor = at + word.length;
    }

    const continues = current !== null && current.group === base && prefix !== "B";
    if (continues && current !== null) {
      current.end = end;
      current.score += entity.score;
      current.count += 1;
    } else {
      flush();
      current = { group: base, score: entity.score, start, end, count: 1 };
    }
  }
  flush();
  return out;
}

function repairSpans(raw: string, spans: readonly Span[], anchorScore: number): Span[] {
  let kept = spans.filter((span) => span.score >= anchorScore).map(copySpan);
  const candidates = spans.filter((span) => span.score >= EXTEND_SCORE && span.score < anchorScore).map(copySpan);

  // Hard cap on the convergence loop. Real text converges in O(maxLabelLen)
  // iterations; rows with many same-label single-token fragments can otherwise
  // make rescue + bridge keep flipping. 32 is well above any healthy run and
  // bounds the pathological case to a few milliseconds.
  const MAX_ITERS = 32;
  let iters = 0;
  let changed = true;
  while (changed && iters < MAX_ITERS) {
    changed = false;
    iters++;

    for (let i = candidates.length - 1; i >= 0; i--) {
      const candidate = candidates[i];
      if (kept.some((span) => canBridge(raw, candidate, span))) {
        kept.push(candidate);
        candidates.splice(i, 1);
        changed = true;
      }
    }

    const merged = mergeAdjacentConnectors(raw, kept);
    const didMerge =
      merged.length !== kept.length ||
      merged.some((span, index) => span.start !== kept[index]?.start || span.end !== kept[index]?.end);
    if (didMerge) {
      changed = true;
    }
    kept = merged;

    for (let i = 0; i < kept.length; i++) {
      const repaired = rescueCapitalizedParticles(raw, kept[i]);
      if (repaired.start !== kept[i].start || repaired.end !== kept[i].end) {
        kept[i] = repaired;
        changed = true;
      }
    }
  }

  return kept
    .map((span) => ({ ...span, text: raw.slice(span.start, span.end) }))
    .sort((a, b) => a.start - b.start || a.end - b.end);
}

function copySpan(span: Span): Span {
  return { ...span };
}

/**
 * True when the character at `idx` is a lone initial — a single capital letter
 * with a non-letter to its left (the "F" in "John F."). An initial's trailing
 * dot is name-internal; a dot after a full word is a sentence boundary.
 */
function isInitialChar(raw: string, idx: number): boolean {
  const c = raw[idx];
  if (c === undefined || !/\p{Lu}/u.test(c)) return false;
  const prev = raw[idx - 1];
  return prev === undefined || !/\p{L}/u.test(prev);
}

function canBridge(raw: string, a: Span, b: Span): boolean {
  if (a.label !== b.label) return false;
  const [left, right] = a.start <= b.start ? [a, b] : [b, a];
  const gap = raw.slice(left.end, right.start);
  if (!CONNECTOR_RE.test(gap)) return false;
  // A period bridges fragments only across an initial ("J." + "R."); a period
  // after a full word ("Garcia." + "I") is a sentence boundary, not a name.
  if (gap.includes(".") && !isInitialChar(raw, left.end - 1)) return false;
  return true;
}

function mergeAdjacentConnectors(raw: string, spans: readonly Span[]): Span[] {
  const merged: Span[] = [];
  for (const span of [...spans].sort((a, b) => a.start - b.start || a.end - b.end)) {
    const previous = merged[merged.length - 1];
    if (previous !== undefined && canBridge(raw, previous, span)) {
      merged[merged.length - 1] = {
        ...previous,
        end: Math.max(previous.end, span.end),
        score: Math.max(previous.score, span.score),
        text: raw.slice(previous.start, Math.max(previous.end, span.end)),
      };
    } else {
      merged.push(copySpan(span));
    }
  }
  return merged;
}

function rescueCapitalizedParticles(raw: string, span: Span): Span {
  if (!PERSON_LABELS.has(span.label)) return span;

  let start = span.start;
  let end = span.end;
  const left = LEFT_PARTICLE_RE.exec(raw.slice(0, start));
  // Extend left across the connector, but let a period through only when the
  // particle is a one-letter initial ("J.", "R."), never a word ("Dr.", "St.").
  if (left !== null && CONNECTOR_RE.test(left[2]) && (!left[2].includes(".") || left[1].length === 1)) {
    start -= left[0].length;
  }

  const right = RIGHT_PARTICLE_RE.exec(raw.slice(end));
  // Never extend right across a period: it always crosses a sentence boundary
  // ("Garcia. I", "Chen. After"). Trailing initials are reached via space.
  if (right !== null && !right[1].includes(".")) {
    end += right[0].length;
  }

  return start === span.start && end === span.end ? span : { ...span, start, end, text: raw.slice(start, end) };
}

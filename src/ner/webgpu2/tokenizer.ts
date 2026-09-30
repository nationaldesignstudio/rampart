/**
 * BERT WordPiece tokenizer for the webgpu2 backend.
 *
 * A dependency-free port of the exact path transformers.js 3.x takes for this
 * model's `tokenizer.json` (BertNormalizer → BertPreTokenizer → WordPiece, with
 * added tokens split out first). It must agree token-for-token with the
 * transformers.js tokenizer: detectNer sizes windows by `countTokens` and
 * recovers char offsets by walking `tokenize` output, and the model was
 * evaluated on those ids. `__tests__/webgpu2-tokenizer.test.ts` holds the two
 * implementations against each other.
 */

// Same character class transformers.js uses (and the Rust tokenizer's BERT
// pre-tokenizer): Unicode punctuation plus every ASCII non-alphanumeric symbol.
const PUNCTUATION = "\\p{P}\\u0021-\\u002F\\u003A-\\u0040\\u005B-\\u0060\\u007B-\\u007E";
const PRE_TOKENIZE_RE = new RegExp(`[^\\s${PUNCTUATION}]+|[${PUNCTUATION}]`, "gu");
// Kept verbatim, operator-precedence quirk and all: transformers.js tests
// `^\p{Cc}` OR `\p{Cf}` OR `\p{Co}` OR `\p{Cs}$`, and we must drop the same chars.
const CONTROL_RE = /^\p{Cc}|\p{Cf}|\p{Co}|\p{Cs}$/u;
const WHITESPACE_RE = /^\s$/;
const NONSPACING_MARK_RE = /\p{Mn}/gu;

interface TokenizerJson {
  readonly added_tokens: ReadonlyArray<{ readonly id: number; readonly content: string; readonly special: boolean }>;
  readonly model: {
    readonly vocab: Readonly<Record<string, number>>;
    readonly unk_token: string;
    readonly continuing_subword_prefix?: string;
    readonly max_input_chars_per_word?: number;
  };
}

function isChineseChar(cp: number): boolean {
  return (
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x20000 && cp <= 0x2a6df) ||
    (cp >= 0x2a700 && cp <= 0x2b73f) ||
    (cp >= 0x2b740 && cp <= 0x2b81f) ||
    (cp >= 0x2b820 && cp <= 0x2ceaf) ||
    (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0x2f800 && cp <= 0x2fa1f)
  );
}

function isControl(char: string): boolean {
  if (char === "\t" || char === "\n" || char === "\r") return false;
  return CONTROL_RE.test(char);
}

/** BertNormalizer with clean_text, handle_chinese_chars, lowercase (+ accent strip). */
function normalize(text: string): string {
  let cleaned = "";
  for (const char of text) {
    // charCodeAt, not codePointAt: transformers.js inspects the first UTF-16 unit.
    const cp = char.charCodeAt(0);
    if (cp === 0 || cp === 0xfffd || isControl(char)) continue;
    cleaned += WHITESPACE_RE.test(char) ? " " : char;
  }
  let spaced = "";
  for (let i = 0; i < cleaned.length; i++) {
    const char = cleaned[i];
    spaced += isChineseChar(char.charCodeAt(0)) ? ` ${char} ` : char;
  }
  return spaced.toLowerCase().normalize("NFD").replace(NONSPACING_MARK_RE, "");
}

export interface BertTokenizer {
  /** WordPiece ids for `text`, without [CLS]/[SEP]. */
  encode(text: string): number[];
  /** WordPiece token strings for `text`, without [CLS]/[SEP]. */
  tokenize(text: string): string[];
  /** The vocab entry for `id` (what a single-id decode yields for non-special ids). */
  idToToken(id: number): string;
  /** True for [PAD]/[UNK]/[CLS]/[SEP]/[MASK]: the ids a skip-special decode empties. */
  isSpecial(id: number): boolean;
  readonly clsId: number;
  readonly sepId: number;
}

export function createBertTokenizer(json: TokenizerJson): BertTokenizer {
  const vocab = new Map<string, number>(Object.entries(json.model.vocab));
  const idToToken: string[] = [];
  for (const [token, id] of vocab) idToToken[id] = token;
  const prefix = json.model.continuing_subword_prefix ?? "##";
  const maxChars = json.model.max_input_chars_per_word ?? 100;
  const unk = json.model.unk_token;

  const added = new Map<string, number>();
  const special = new Set<number>();
  for (const t of json.added_tokens) {
    added.set(t.content, t.id);
    vocab.set(t.content, t.id);
    idToToken[t.id] = t.content;
    if (t.special) special.add(t.id);
  }
  // Added tokens are matched longest-first at each position, before
  // normalization — the same split transformers.js' DictionarySplitter does.
  const addedByLength = [...added.keys()].sort((a, b) => b.length - a.length);
  const addedFirstChars = new Set(addedByLength.map((t) => t[0]));

  function splitAdded(text: string): string[] {
    const out: string[] = [];
    let start = 0;
    let i = 0;
    while (i < text.length) {
      const match = addedFirstChars.has(text[i]) ? addedByLength.find((t) => text.startsWith(t, i)) : undefined;
      if (match === undefined) {
        i++;
        continue;
      }
      if (i > start) out.push(text.slice(start, i));
      out.push(match);
      i += match.length;
      start = i;
    }
    if (start < text.length) out.push(text.slice(start));
    return out;
  }

  function wordPiece(word: string, out: string[]): void {
    const chars = [...word];
    if (chars.length > maxChars) {
      out.push(unk);
      return;
    }
    const pieces: string[] = [];
    let start = 0;
    while (start < chars.length) {
      let end = chars.length;
      let found: string | null = null;
      while (start < end) {
        let sub = chars.slice(start, end).join("");
        if (start > 0) sub = prefix + sub;
        if (vocab.has(sub)) {
          found = sub;
          break;
        }
        end--;
      }
      if (found === null) {
        out.push(unk);
        return;
      }
      pieces.push(found);
      start = end;
    }
    for (const p of pieces) out.push(p);
  }

  function tokenize(text: string): string[] {
    const out: string[] = [];
    for (const section of splitAdded(text)) {
      if (section.length === 0) continue;
      if (added.has(section)) {
        out.push(section);
        continue;
      }
      const normalized = normalize(section);
      if (normalized.length === 0) continue;
      for (const word of normalized.trim().match(PRE_TOKENIZE_RE) ?? []) wordPiece(word, out);
    }
    return out;
  }

  const unkId = vocab.get(unk) ?? 0;
  return {
    tokenize,
    encode: (text) => tokenize(text).map((t) => vocab.get(t) ?? unkId),
    idToToken: (id) => idToToken[id] ?? unk,
    isSpecial: (id) => special.has(id),
    clsId: vocab.get("[CLS]") ?? 2,
    sepId: vocab.get("[SEP]") ?? 3,
  };
}

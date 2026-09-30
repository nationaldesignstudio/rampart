import { readFileSync } from "node:fs";
import { join } from "node:path";
import { AutoTokenizer, env } from "@huggingface/transformers";
import { describe, expect, it } from "vitest";
import { PUBLIC_E2E_CASES } from "../eval/public-cases";
import { createBertTokenizer } from "../src/ner/webgpu2/tokenizer";

const MODEL_DIR = join(import.meta.dirname, "..", "model");

// Inputs chosen to hit every normalizer/pre-tokenizer branch: accents, CJK,
// control and format chars, added tokens inside text, premask sentinels,
// over-long words, emoji (surrogate pairs), and odd whitespace.
const EDGE_CASES = [
  "José Ångström lives at 12 Rue de l'Église, Montréal.",
  "我的名字是王小明 and my email is wang@example.com",
  "tab\tnew\nline\rcarriage\u0000nul​zwsp­soft�repl",
  "Call [SSN] or [CREDIT_CARD] then [MASK][CLS]x[SEP] [UNK]",
  "a".repeat(150) + " short",
  "emoji 👩‍💻 test 🎉 and 𠀀 rare cjk",
  "  leading and trailing   ",
  "",
  "don't re-enter O'Brien-Smith's i.e. U.S.A. #hash $5.00 ~tilde~",
  "ΑΒΓ Ελληνικά Кириллица العربية हिन्दी",
];

describe("webgpu2 tokenizer parity with transformers.js", () => {
  it("matches token strings and ids exactly", async () => {
    env.allowRemoteModels = false;
    env.localModelPath = join(MODEL_DIR, "..");
    const reference = await AutoTokenizer.from_pretrained("model");
    const ours = createBertTokenizer(JSON.parse(readFileSync(join(MODEL_DIR, "tokenizer.json"), "utf8")));
    const texts = [...EDGE_CASES, ...PUBLIC_E2E_CASES.map((c) => c.input)];
    for (const text of texts) {
      expect(ours.tokenize(text), text).toEqual(reference.tokenize(text));
      expect(ours.encode(text), text).toEqual(reference.encode(text, { add_special_tokens: false }));
    }
  });
});

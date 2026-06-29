import { describe, expect, it } from "vitest";
import { detectNer, NER_TOKEN_BUDGET, type TokenClassifier, type TokenCounter } from "../src/ner/classifier";

// A stand-in tokenizer for the mocks: one token per whitespace-delimited word.
// detectNer windows by this count, so it drives chunking deterministically.
const countWords: TokenCounter = (text) => (text.match(/\S+/g) ?? []).length;

describe("detectNer span repair", () => {
  it("runs inference on a hyphen-normalized copy and repairs hyphenated person spans", async () => {
    let modelInput = "";
    const classifier: TokenClassifier = async (text) => {
      modelInput = text;
      return [
        { entity_group: "GIVEN_NAME", score: 0.55, start: 11, end: 16, word: "Thanh" },
        { entity_group: "GIVEN_NAME", score: 0.22, start: 17, end: 23, word: "Nghiem" },
        { entity_group: "GIVEN_NAME", score: 0.45, start: 24, end: 28, word: "Quoc" },
        { entity_group: "GIVEN_NAME", score: 0.21, start: 29, end: 32, word: "Bao" },
      ];
    };

    const text = "My name is Thanh-Nghiem Quoc-Bao.";
    const spans = await detectNer(text, classifier);

    expect(modelInput).toBe("My name is Thanh Nghiem Quoc Bao.");
    expect(spans).toEqual([
      {
        start: 11,
        end: 32,
        label: "GIVEN_NAME",
        score: 0.55,
        source: "ner",
        text: "Thanh-Nghiem Quoc-Bao",
      },
    ]);
  });

  it("recovers raw offsets for folded BIO tokens over accented text", async () => {
    // transformers.js emits raw BIO tokens whose `word` is already accent-folded
    // (lowercased, diacritics stripped) and carries no usable char offsets. The
    // runtime must still redact the accented name at its true raw offsets — a
    // naive indexOf against the unfolded text fails on every accented character.
    const classifier: TokenClassifier = async () =>
      [
        { entity: "B-GIVEN_NAME", score: 0.99, word: "genevieve", start: 0, end: 0 },
        { entity: "I-GIVEN_NAME", score: 0.99, word: "muller", start: 0, end: 0 },
      ] as Awaited<ReturnType<TokenClassifier>>;

    const spans = await detectNer("Bonjour Geneviève Müller, votre dossier est prêt.", classifier);

    expect(spans).toEqual([
      {
        start: 8,
        end: 24,
        label: "GIVEN_NAME",
        score: 0.99,
        source: "ner",
        text: "Geneviève Müller",
      },
    ]);
  });

  it("extends person spans over short capitalized particles", async () => {
    const classifier: TokenClassifier = async () => [
      { entity_group: "GIVEN_NAME", score: 0.61, start: 16, end: 21, word: "Croix" },
    ];

    const spans = await detectNer("Applicant De La Croix applied.", classifier);

    expect(spans.map((span) => span.text)).toEqual(["De La Croix"]);
  });

  it("joins initials across periods but not full words across sentence boundaries", async () => {
    // The model tags only the surname; particle rescue must grow left across the
    // "J. R. R." initials (a period after a single letter is name-internal).
    const initials: TokenClassifier = async () => [
      { entity_group: "GIVEN_NAME", score: 0.9, start: 9, end: 16, word: "Tolkien" },
    ];
    const initialsSpans = await detectNer("J. R. R. Tolkien wrote the book.", initials);
    expect(initialsSpans.map((s) => s.text)).toEqual(["J. R. R. Tolkien"]);

    // A period after a full word ends a sentence: the following capitalized word
    // must NOT be swallowed into the name span.
    const boundary: TokenClassifier = async () => [
      { entity_group: "GIVEN_NAME", score: 0.9, start: 0, end: 12, word: "Maria Garcia" },
    ];
    const boundarySpans = await detectNer("Maria Garcia. I think she left.", boundary);
    expect(boundarySpans.map((s) => s.text)).toEqual(["Maria Garcia"]);
  });

  it("does not merge two distinct names across a sentence-ending period", async () => {
    const classifier: TokenClassifier = async () => [
      { entity_group: "GIVEN_NAME", score: 0.9, start: 11, end: 16, word: "Maria" },
      { entity_group: "GIVEN_NAME", score: 0.9, start: 18, end: 21, word: "Bob" },
    ];

    const spans = await detectNer("We emailed Maria. Bob replied later.", classifier);

    expect(spans.map((s) => s.text)).toEqual(["Maria", "Bob"]);
  });

  it("windows long input by token budget so every model call fits, and the tail is not lost", async () => {
    const inputs: string[] = [];
    const classifier: TokenClassifier = async (text) => {
      inputs.push(text);
      const at = text.indexOf("Jordan Lee");
      return at < 0
        ? []
        : [{ entity_group: "GIVEN_NAME", score: 0.95, start: at, end: at + "Jordan Lee".length, word: "Jordan Lee" }];
    };
    classifier.countTokens = countWords;

    // Many more tokens than one window holds; the name sits past the first window.
    const longText = `${"word ".repeat(NER_TOKEN_BUDGET * 2)}Jordan Lee`;
    const spans = await detectNer(longText, classifier);

    expect(inputs.length).toBeGreaterThan(1); // it was actually chunked
    for (const input of inputs) {
      expect(countWords(input)).toBeLessThanOrEqual(NER_TOKEN_BUDGET);
    }

    // The name past the first window is redacted at its true offset, not dropped.
    const jordan = spans.find((s) => s.text === "Jordan Lee");
    expect(jordan).toBeDefined();
    expect(longText.slice(jordan!.start, jordan!.end)).toBe("Jordan Lee");
  });

  it("recovers an entity straddling a token-window seam via the overlap, exactly once", async () => {
    const classifier: TokenClassifier = async (text) => {
      const at = text.indexOf("Jordan Lee");
      return at < 0
        ? []
        : [{ entity_group: "GIVEN_NAME", score: 0.95, start: at, end: at + "Jordan Lee".length, word: "Jordan Lee" }];
    };
    classifier.countTokens = countWords;

    // Fill exactly one window's worth of words so "Jordan" is the last word of
    // window one and "Lee" only appears in the overlapping window two — the name
    // straddles the seam and is whole only in window two.
    const filler = Array(NER_TOKEN_BUDGET - 1).fill("word").join(" ");
    const text = `${filler} Jordan Lee and the rest of the message continues.`;
    const spans = await detectNer(text, classifier);

    const jordan = spans.filter((s) => s.text === "Jordan Lee");
    expect(jordan).toHaveLength(1);
    expect(text.slice(jordan[0].start, jordan[0].end)).toBe("Jordan Lee");
  });

  it("hard-splits an over-budget unbroken token so no window exceeds the budget", async () => {
    // One whitespace-delimited "word" longer than a window can ever hold (a hash
    // or base64 blob): it must still be char-split so the model never sees a
    // sequence past its budget. Such blobs carry no NER entities — the safety
    // property under test is purely that every window fits.
    const countChars: TokenCounter = (text) => text.length; // one token per char
    const inputs: string[] = [];
    const classifier: TokenClassifier = async (text) => {
      inputs.push(text);
      return [];
    };
    classifier.countTokens = countChars;

    const blob = "x".repeat(NER_TOKEN_BUDGET * 3 + 17); // no whitespace, far over budget
    await detectNer(blob, classifier);

    expect(inputs.length).toBeGreaterThan(1);
    for (const input of inputs) {
      expect(countChars(input)).toBeLessThanOrEqual(NER_TOKEN_BUDGET);
    }
  });

  it("anchors BIO offsets to token index so duplicate digit runs map correctly", async () => {
    const text = "or 555-123-4567. I live at 1234 Maple Street.";
    const tokenize = (input: string) => {
      const infer = input.replaceAll("-", " ");
      const pieces: string[] = [];
      for (const word of infer.split(/\s+/)) {
        if (word.length === 0) continue;
        if (word === "1234") {
          pieces.push("123", "##4");
          continue;
        }
        pieces.push(word);
      }
      return pieces;
    };
    const classifier: TokenClassifier = async (input) => {
      const tokens = tokenize(input);
      const indexOf = (piece: string, from: number) => tokens.findIndex((t, i) => i >= from && t === piece);
      const at = indexOf("at", 0);
      const building123 = indexOf("123", at + 1);
      const building4 = indexOf("##4", building123 + 1);
      const maple = indexOf("maple", building4 + 1);
      const street = indexOf("street", maple + 1);
      return [
        { entity: "B-BUILDING_NUMBER", score: 0.99, index: building123 + 1, word: "123", start: 0, end: 0 },
        { entity: "B-BUILDING_NUMBER", score: 0.99, index: building4 + 1, word: "##4", start: 0, end: 0 },
        { entity: "B-STREET_NAME", score: 0.99, index: maple + 1, word: "maple", start: 0, end: 0 },
        { entity: "I-STREET_NAME", score: 0.99, index: street + 1, word: "street", start: 0, end: 0 },
      ];
    };
    classifier.tokenize = tokenize;

    const spans = await detectNer(text, classifier);
    const building = spans.find((s) => s.label === "BUILDING_NUMBER");
    const street = spans.find((s) => s.label === "STREET_NAME");

    expect(building?.text).toBe("1234");
    expect(street?.text).toBe("Maple Street");
    expect(text.slice(building!.start, building!.end)).toBe("1234");
  });

  it("does not keep isolated below-anchor spans", async () => {
    const classifier: TokenClassifier = async () => [
      { entity_group: "GIVEN_NAME", score: 0.22, start: 11, end: 16, word: "Maybe" },
    ];

    const spans = await detectNer("The word is Maybe here.", classifier);

    expect(spans).toEqual([]);
  });
});

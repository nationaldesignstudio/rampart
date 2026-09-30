/**
 * webgpu2: the shipped q4 Rampart classifier on hand-written WGSL kernels.
 *
 * A drop-in {@link TokenClassifier} that skips transformers.js and ONNX Runtime
 * entirely. It fetches the same `onnx/model_q4.onnx`, `tokenizer.json`, and
 * `config.json` the transformers.js path does, reads the q4 weights straight
 * out of the ONNX file, and runs them on the kernels in `./shaders`. Output
 * rows have the same shape transformers.js' token-classification pipeline
 * emits (`entity`, `score`, `index`, `word`, "O" and specials dropped), so
 * detectNer's offset recovery and span repair are unchanged.
 */

import type { TokenClassifier } from "../classifier";
import { readBertQ4, type ModelConfig } from "./onnx";
import { Webgpu2Runtime } from "./runtime";
import { createBertTokenizer } from "./tokenizer";

export interface Webgpu2Options {
  /**
   * Hugging Face model id (`org/name`, fetched from the Hub's `resolve/main`)
   * or a base URL / absolute path that serves the model directory
   * (`https://…/model`, `/model`, `./model`).
   */
  readonly model: string;
  /** Hub revision when `model` is an id. Defaults to `main`. */
  readonly revision?: string;
}

const CACHE_NAME = "rampart-webgpu2";

function resolveBase(model: string, revision: string): string {
  if (/^(https?:)?\/\//.test(model) || model.startsWith("/") || model.startsWith(".")) {
    return model.endsWith("/") ? model : `${model}/`;
  }
  return `https://huggingface.co/${model}/resolve/${revision}/`;
}

/**
 * Fetch through the Cache API when present, so the 14.7 MB model is downloaded
 * once per origin like the transformers.js path's own browser cache.
 */
async function fetchCached(url: string): Promise<Response> {
  const cache = typeof caches === "undefined" ? null : await caches.open(CACHE_NAME).catch(() => null);
  const hit = await cache?.match(url);
  if (hit) return hit;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`[rampart] webgpu2: fetching ${url} failed (${res.status})`);
  if (cache && !url.startsWith("http://localhost")) await cache.put(url, res.clone()).catch(() => undefined);
  return res;
}

export async function loadWebgpu2Classifier(options: Webgpu2Options): Promise<TokenClassifier> {
  const base = resolveBase(options.model, options.revision ?? "main");
  const [config, tokenizerJson, onnx] = await Promise.all([
    fetchCached(`${base}config.json`).then((r) => r.json() as Promise<ModelConfig>),
    fetchCached(`${base}tokenizer.json`).then((r) => r.json()),
    fetchCached(`${base}onnx/model_q4.onnx`).then((r) => r.arrayBuffer()),
  ]);
  const tokenizer = createBertTokenizer(tokenizerJson);
  const runtime = await Webgpu2Runtime.create(readBertQ4(new Uint8Array(onnx), config));
  const id2label = config.id2label;
  // [CLS] + content + [SEP] must fit the position table. detectNer's windows
  // stay well under this; anything longer is truncated like the pipeline would.
  const maxContent = runtime.maxTokens - 2;

  const classifier: TokenClassifier = async (text) => {
    const content = tokenizer.encode(text).slice(0, maxContent);
    const ids = [tokenizer.clsId, ...content, tokenizer.sepId];
    const { labels, scores } = await runtime.run(ids);
    const rows = [];
    for (let j = 0; j < ids.length; j++) {
      const entity = id2label[labels[j]] ?? `LABEL_${labels[j]}`;
      if (entity === "O" || tokenizer.isSpecial(ids[j])) continue;
      rows.push({ entity, score: scores[j], index: j, word: tokenizer.idToToken(ids[j]) });
    }
    return rows;
  };
  classifier.countTokens = (text) => tokenizer.encode(text).length;
  classifier.tokenize = (text) => tokenizer.tokenize(text);
  return classifier;
}

export { Webgpu2Runtime } from "./runtime";
export { createBertTokenizer } from "./tokenizer";
export { readBertQ4 } from "./onnx";

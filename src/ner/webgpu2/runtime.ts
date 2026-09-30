/**
 * WebGPU runtime for the webgpu2 backend.
 *
 * Everything that can be done once is done at load: weights uploaded in their
 * q4 form, all activation buffers allocated at the model's 512-position limit,
 * every pipeline compiled and every bind group built. A call then costs two
 * small `writeBuffer`s (ids, token count), one command buffer holding 44
 * dispatches in a single compute pass, and one `mapAsync` of T×8 bytes.
 *
 * One pass is enough for correctness: WebGPU makes each dispatch its own
 * usage scope, so storage writes from one dispatch are visible to the next.
 */

import type { BertQ4Weights, Q4Linear, U8Table } from "./onnx";
import {
  attentionShader,
  classifierShader,
  embedShader,
  gemmShader,
  layerNormShader,
  type ShaderDims,
} from "./shaders";

export interface Webgpu2Output {
  /** Argmax label id per position (including [CLS]/[SEP]). */
  readonly labels: Uint32Array;
  /** Softmax probability of the argmax label per position. */
  readonly scores: Float32Array;
}

interface Step {
  readonly pipeline: GPUComputePipeline;
  readonly bind: GPUBindGroup;
  /** Workgroup counts for a sequence of `t` tokens. */
  readonly grid: (t: number) => [number, number] | [number, number, number];
}

const ceil = (a: number, b: number): number => Math.ceil(a / b);

export class Webgpu2Runtime {
  readonly maxTokens: number;
  readonly numLabels: number;
  #device: GPUDevice;
  #steps: Step[] = [];
  #owned: GPUBuffer[] = [];
  #ids!: GPUBuffer;
  #params!: GPUBuffer;
  #result!: GPUBuffer;
  #logits!: GPUBuffer;
  #staging!: GPUBuffer;
  #chain: Promise<unknown> = Promise.resolve();
  #lost: string | null = null;

  static async create(weights: BertQ4Weights, device?: GPUDevice): Promise<Webgpu2Runtime> {
    let dev = device;
    if (!dev) {
      const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
      if (!gpu) throw new Error("[rampart] webgpu2: WebGPU is not available in this environment");
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("[rampart] webgpu2: no WebGPU adapter");
      // The word-embedding table is 7.6 MB packed; default binding limits
      // (128 MiB) cover it, so the default device is enough.
      dev = await adapter.requestDevice();
    }
    // WebGPU validation errors are otherwise silent: an invalid bind group just
    // turns its dispatch into a no-op and the model emits plausible garbage.
    dev.addEventListener("uncapturederror", (e) => {
      console.error("[rampart] webgpu2:", (e as GPUUncapturedErrorEvent).error.message);
    });
    const rt = new Webgpu2Runtime(dev, weights);
    // Mobile browsers drop GPU devices on backgrounding and memory pressure.
    // Fail calls loudly rather than hang on a mapAsync that never resolves.
    void dev.lost.then((info) => {
      rt.#lost = `device lost (${info.reason}): ${info.message}`;
    });
    await rt.#checkCompilation();
    return rt;
  }

  #modules: GPUShaderModule[] = [];

  private constructor(device: GPUDevice, w: BertQ4Weights) {
    this.#device = device;
    this.maxTokens = w.maxPositions;
    this.numLabels = w.numLabels;
    const H = w.hidden;
    const T = this.maxTokens;
    const dims: ShaderDims = {
      hidden: H,
      heads: w.heads,
      intermediate: w.intermediate,
      numLabels: w.numLabels,
      vocab: w.wordEmb.rows,
      eps: w.layerNormEps,
    };

    const S = GPUBufferUsage.STORAGE;
    this.#ids = this.#buffer(T * 4, S | GPUBufferUsage.COPY_DST, "ids");
    this.#params = this.#buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "params");
    const h = this.#buffer(T * H * 4, S, "hidden");
    const qkv = this.#buffer(T * 3 * H * 4, S, "qkv");
    const ctx = this.#buffer(T * H * 4, S, "ctx");
    const inter = this.#buffer(T * w.intermediate * 4, S, "ffn");
    // FFN-out walks K=1536, 4× the other GEMMs, over only 6 column tiles per
    // row tile. Splitting K in two halves that serial chain (the LayerNorm after
    // it sums the halves). Measured on M2 Max: split 2 ≈ split 4 (±0.1 ms) and
    // ~0.3 ms faster than no split at T≈30, so take the one that launches fewer
    // workgroups — this has to run well on phone GPUs with a handful of cores.
    const SPLIT = 2;
    const TM = 32;
    const preLn = this.#buffer(SPLIT * T * H * 4, S, "pre_ln");
    this.#result = this.#buffer(T * 8, S | GPUBufferUsage.COPY_SRC, "result");
    this.#logits = this.#buffer(T * w.numLabels * 4, S | GPUBufferUsage.COPY_SRC, "logits");
    this.#staging = this.#buffer(T * w.numLabels * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, "staging");

    const pipe = (code: string, label: string): GPUComputePipeline => {
      const module = device.createShaderModule({ code, label });
      this.#modules.push(module);
      return device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" }, label });
    };
    const step = (pipeline: GPUComputePipeline, buffers: GPUBuffer[], grid: Step["grid"]): void => {
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      this.#steps.push({ pipeline, bind, grid });
    };
    const lin = (l: Q4Linear, name: string) => ({
      q: this.#upload(l.q, name + ".q4"),
      s: this.#upload(l.scales, name + ".scales"),
      b: this.#upload(l.bias, name + ".bias"),
    });
    const vecs = (label: string, ...parts: Float32Array[]): GPUBuffer => {
      const all = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
      let o = 0;
      for (const p of parts) {
        all.set(p, o);
        o += p.length;
      }
      return this.#upload(all, label);
    };
    const qp = (t: U8Table): [number, number] => [t.scale, t.zeroPoint];

    const embed = pipe(embedShader(dims, { word: qp(w.wordEmb), pos: qp(w.posEmb), type: qp(w.typeEmb) }), "embed");
    const gemmQkv = pipe(gemmShader(H, 3 * H), "gemm_qkv");
    const attention = pipe(attentionShader(dims), "attention");
    const gemmOut = pipe(gemmShader(H, H, { residual: true, maxTokens: T }), "gemm_out_residual");
    const ffnIn = pipe(gemmShader(H, w.intermediate, { gelu: true, maxTokens: T }), "gemm_ffn_gelu");
    const ffnOut = pipe(gemmShader(w.intermediate, H, { residual: true, splitK: SPLIT, maxTokens: T }), "gemm_ffn_out_residual");
    const layerNorm = pipe(layerNormShader(dims, 1, T), "layer_norm");
    const layerNormSplit = pipe(layerNormShader(dims, SPLIT, T), "layer_norm_split");
    const classify = pipe(classifierShader(dims), "classifier");

    step(
      embed,
      [
        this.#ids,
        this.#upload(w.wordEmb.q, "emb.word"),
        this.#upload(w.posEmb.q, "emb.pos"),
        this.#upload(w.typeEmb.q, "emb.type"),
        vecs("emb.ln", w.embLn.gamma, w.embLn.beta),
        h,
      ],
      (t) => [t, 1],
    );
    w.layers.forEach((layer, i) => {
      const a = lin(layer.qkv, `l${i}.qkv`);
      step(gemmQkv, [h, a.q, a.s, a.b, qkv, this.#params], (t) => [(3 * H) / 64, ceil(t, TM)]);
      step(attention, [qkv, ctx, this.#params], (t) => [ceil(t, 16), w.heads]);
      const o = lin(layer.attnOut, `l${i}.attn_out`);
      step(gemmOut, [ctx, o.q, o.s, o.b, preLn, this.#params, h], (t) => [H / 64, ceil(t, TM)]);
      step(layerNorm, [preLn, vecs(`l${i}.ln1`, layer.ln1.gamma, layer.ln1.beta), h], (t) => [t, 1]);
      const f1 = lin(layer.ffnIn, `l${i}.ffn_in`);
      step(ffnIn, [h, f1.q, f1.s, f1.b, inter, this.#params], (t) => [w.intermediate / 64, ceil(t, TM)]);
      const f2 = lin(layer.ffnOut, `l${i}.ffn_out`);
      step(ffnOut, [inter, f2.q, f2.s, f2.b, preLn, this.#params, h], (t) => [H / 64, ceil(t, TM), SPLIT]);
      step(layerNormSplit, [preLn, vecs(`l${i}.ln2`, layer.ln2.gamma, layer.ln2.beta), h], (t) => [t, 1]);
    });
    const c = lin(w.classifier, "classifier");
    // Row-per-workgroup kernels dispatch exactly T workgroups, so they bind no params
    // (an `auto` layout drops unused bindings, which would reject the entry).
    step(classify, [h, c.q, c.s, c.b, this.#result, this.#logits], (t) => [t, 1]);
  }

  async #checkCompilation(): Promise<void> {
    for (const m of this.#modules) {
      const info = await m.getCompilationInfo();
      const errors = info.messages.filter((x) => x.type === "error");
      if (errors.length) {
        throw new Error(
          `[rampart] webgpu2: WGSL compilation failed in ${m.label}:\n` +
            errors.map((e) => `  ${e.lineNum}:${e.linePos} ${e.message}`).join("\n"),
        );
      }
    }
  }

  #buffer(size: number, usage: number, label: string): GPUBuffer {
    const b = this.#device.createBuffer({ size: Math.ceil(size / 16) * 16, usage, label });
    this.#owned.push(b);
    return b;
  }

  #upload(data: Uint32Array | Float32Array, label: string): GPUBuffer {
    const b = this.#buffer(data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label);
    this.#device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  }

  /** Classify one sequence of token ids ([CLS] … [SEP], at most `maxTokens`). */
  run(ids: ArrayLike<number>): Promise<Webgpu2Output> {
    return this.#serialize(async () => {
      const raw = await this.#execute(ids, false);
      const u = new Uint32Array(raw);
      const f = new Float32Array(raw);
      const t = ids.length;
      const labels = new Uint32Array(t);
      const scores = new Float32Array(t);
      for (let i = 0; i < t; i++) {
        labels[i] = u[2 * i];
        scores[i] = f[2 * i + 1];
      }
      return { labels, scores };
    });
  }

  /** Raw `[T, numLabels]` logits, for parity checks against the ONNX reference. */
  logits(ids: ArrayLike<number>): Promise<Float32Array> {
    return this.#serialize(async () => new Float32Array(await this.#execute(ids, true)));
  }

  // One staging buffer means one map in flight; calls queue behind each other
  // rather than tripping "buffer already has an outstanding map pending".
  #serialize<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.#chain.then(fn, fn);
    this.#chain = next.catch(() => undefined);
    return next;
  }

  async #execute(ids: ArrayLike<number>, wantLogits: boolean): Promise<ArrayBuffer> {
    if (this.#lost) throw new Error(`[rampart] webgpu2: ${this.#lost}`);
    const t = ids.length;
    if (t === 0) return new ArrayBuffer(0);
    if (t > this.maxTokens) throw new Error(`[rampart] webgpu2: ${t} tokens exceeds ${this.maxTokens}`);
    const dev = this.#device;
    dev.queue.writeBuffer(this.#ids, 0, Uint32Array.from(ids));
    dev.queue.writeBuffer(this.#params, 0, new Uint32Array([t, 0, 0, 0]));

    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const s of this.#steps) {
      pass.setPipeline(s.pipeline);
      pass.setBindGroup(0, s.bind);
      const [x, y, z = 1] = s.grid(t);
      pass.dispatchWorkgroups(x, y, z);
    }
    pass.end();
    const bytes = wantLogits ? t * this.numLabels * 4 : t * 8;
    enc.copyBufferToBuffer(wantLogits ? this.#logits : this.#result, 0, this.#staging, 0, Math.ceil(bytes / 4) * 4);
    dev.queue.submit([enc.finish()]);

    await this.#staging.mapAsync(GPUMapMode.READ, 0, bytes);
    const out = this.#staging.getMappedRange(0, bytes).slice(0);
    this.#staging.unmap();
    return out;
  }

  destroy(): void {
    for (const b of this.#owned) b.destroy();
    this.#owned = [];
  }
}

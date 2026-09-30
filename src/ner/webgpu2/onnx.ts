/**
 * Minimal ONNX reader for the shipped q4 BERT token classifier.
 *
 * Reads `model_q4.onnx` directly — no conversion step, no second artifact — and
 * returns the tensors the webgpu2 kernels consume, still in their quantized form:
 *
 *  - every Linear is a `com.microsoft:MatMulNBits` (bits=4, block_size=32, no
 *    zero point → implicit 8). Its `[N, K/32, 16]` uint8 payload is uploaded
 *    as-is: each 16-byte block is four little-endian u32 words, nibble `k` of a
 *    block sitting at bits `4*(k%8)` of word `k/8`. The shader dequantizes in
 *    registers, so the GPU holds 4.5 bits/weight, same as the file.
 *  - the three embedding tables are per-tensor uint8 `DequantizeLinear`, also
 *    uploaded packed (four bytes per u32).
 *  - biases and LayerNorm params are fp32.
 *
 * Linears are located by the graph's node *outputs* (`.../query/MatMul_output_0`),
 * not by initializer names, which the exporter numbered opaquely.
 */

export interface Q4Linear {
  readonly n: number;
  readonly k: number;
  /** `[N, K/32, 4]` u32 words of packed 4-bit weights. */
  readonly q: Uint32Array;
  /** `[N, K/32]` fp32 block scales. */
  readonly scales: Float32Array;
  readonly bias: Float32Array;
}

export interface U8Table {
  readonly rows: number;
  readonly cols: number;
  /** Row-major uint8 values, four per u32. */
  readonly q: Uint32Array;
  readonly scale: number;
  readonly zeroPoint: number;
}

export interface BertLayer {
  /** Fused Q|K|V projection: N = 3 * hidden. */
  readonly qkv: Q4Linear;
  readonly attnOut: Q4Linear;
  readonly ln1: { readonly gamma: Float32Array; readonly beta: Float32Array };
  readonly ffnIn: Q4Linear;
  readonly ffnOut: Q4Linear;
  readonly ln2: { readonly gamma: Float32Array; readonly beta: Float32Array };
}

export interface BertQ4Weights {
  readonly hidden: number;
  readonly heads: number;
  readonly intermediate: number;
  readonly numLabels: number;
  readonly maxPositions: number;
  readonly layerNormEps: number;
  readonly wordEmb: U8Table;
  readonly posEmb: U8Table;
  readonly typeEmb: U8Table;
  readonly embLn: { readonly gamma: Float32Array; readonly beta: Float32Array };
  readonly layers: readonly BertLayer[];
  readonly classifier: Q4Linear;
}

// --- protobuf wire format ---------------------------------------------------

class Reader {
  pos: number;
  constructor(
    readonly buf: Uint8Array,
    start = 0,
    readonly end = buf.length,
  ) {
    this.pos = start;
  }
  varint(): number {
    // Values here (dims, lengths, enum tags) fit in 2^53; accumulate as a double.
    let result = 0;
    let mul = 1;
    for (;;) {
      const b = this.buf[this.pos++];
      result += (b & 0x7f) * mul;
      if (b < 0x80) return result;
      mul *= 128;
    }
  }
  /** Yields [field, wireType] and leaves `pos` at the payload. */
  *fields(): Generator<[number, number]> {
    while (this.pos < this.end) {
      const tag = this.varint();
      yield [Math.floor(tag / 8), tag & 7];
    }
  }
  /** A reader over the next length-delimited payload; advances past it. */
  sub(): Reader {
    const len = this.varint();
    const r = new Reader(this.buf, this.pos, this.pos + len);
    this.pos += len;
    return r;
  }
  bytes(): Uint8Array {
    const len = this.varint();
    const out = this.buf.subarray(this.pos, this.pos + len);
    this.pos += len;
    return out;
  }
  skip(wire: number): void {
    if (wire === 0) this.varint();
    else if (wire === 1) this.pos += 8;
    else if (wire === 2) {
      // Read the length first: `this.pos += this.varint()` would add it to the
      // position from *before* the length varint was consumed.
      const len = this.varint();
      this.pos += len;
    }
    else if (wire === 5) this.pos += 4;
    else throw new Error(`onnx: unsupported wire type ${wire}`);
  }
}

const decoder = new TextDecoder();

interface RawTensor {
  name: string;
  dims: number[];
  dataType: number;
  raw: Uint8Array | null;
  floats: number[];
  ints: number[];
}

// TensorProto field numbers: dims=1, data_type=2, float_data=4, int32_data=5, name=8, raw_data=9.
function readTensor(bytes: Uint8Array): RawTensor {
  const r = new Reader(bytes);
  const t: RawTensor = { name: "", dims: [], dataType: 0, raw: null, floats: [], ints: [] };
  for (const [field, wire] of r.fields()) {
    if (field === 1 && wire === 0) t.dims.push(r.varint());
    else if (field === 1 && wire === 2) {
      const p = r.sub();
      while (p.pos < p.end) t.dims.push(p.varint());
    } else if (field === 2) t.dataType = r.varint();
    else if (field === 4 && wire === 2) {
      const b = r.bytes();
      const f = new Float32Array(b.slice().buffer);
      for (const v of f) t.floats.push(v);
    } else if (field === 4 && wire === 5) {
      t.floats.push(new DataView(r.buf.buffer, r.buf.byteOffset + r.pos, 4).getFloat32(0, true));
      r.pos += 4;
    } else if (field === 5 && wire === 2) {
      const p = r.sub();
      while (p.pos < p.end) t.ints.push(p.varint());
    } else if (field === 5 && wire === 0) t.ints.push(r.varint());
    else if (field === 8) t.name = decoder.decode(r.bytes());
    else if (field === 9) t.raw = r.bytes();
    else r.skip(wire);
  }
  return t;
}

// NodeProto field numbers: input=1, output=2, op_type=4, attribute=5.
interface RawNode {
  inputs: string[];
  outputs: string[];
  opType: string;
}

function readNode(bytes: Uint8Array): RawNode {
  const r = new Reader(bytes);
  const n: RawNode = { inputs: [], outputs: [], opType: "" };
  for (const [field, wire] of r.fields()) {
    if (field === 1) n.inputs.push(decoder.decode(r.bytes()));
    else if (field === 2) n.outputs.push(decoder.decode(r.bytes()));
    else if (field === 4) n.opType = decoder.decode(r.bytes());
    else r.skip(wire);
  }
  return n;
}

function toFloat32(t: RawTensor): Float32Array {
  if (t.dataType !== 1) throw new Error(`onnx: ${t.name} is dtype ${t.dataType}, expected float32`);
  if (t.raw) return new Float32Array(t.raw.slice().buffer);
  return Float32Array.from(t.floats);
}

function toU32Words(bytes: Uint8Array): Uint32Array {
  const padded = new Uint8Array(Math.ceil(bytes.length / 4) * 4);
  padded.set(bytes);
  return new Uint32Array(padded.buffer);
}

function scalar(t: RawTensor): number {
  if (t.raw) {
    if (t.dataType === 1) return new DataView(t.raw.buffer, t.raw.byteOffset, 4).getFloat32(0, true);
    return t.raw[0];
  }
  return t.dataType === 1 ? t.floats[0] : t.ints[0];
}

export interface ModelConfig {
  readonly hidden_size: number;
  readonly num_attention_heads: number;
  readonly num_hidden_layers: number;
  readonly intermediate_size: number;
  readonly max_position_embeddings: number;
  readonly layer_norm_eps: number;
  readonly id2label: Readonly<Record<string, string>>;
}

/** Parse `model_q4.onnx` into the tensors the webgpu2 kernels bind. */
export function readBertQ4(bytes: Uint8Array, config: ModelConfig): BertQ4Weights {
  const tensors = new Map<string, RawTensor>();
  const nodes: RawNode[] = [];
  const model = new Reader(bytes);
  for (const [field, wire] of model.fields()) {
    if (field !== 7) {
      model.skip(wire);
      continue;
    }
    // GraphProto: node=1, initializer=5.
    const g = model.sub();
    for (const [gf, gw] of g.fields()) {
      if (gf === 1) nodes.push(readNode(g.bytes()));
      else if (gf === 5) {
        const t = readTensor(g.bytes());
        tensors.set(t.name, t);
      } else g.skip(gw);
    }
  }

  const get = (name: string): RawTensor => {
    const t = tensors.get(name);
    if (!t) throw new Error(`onnx: missing initializer ${name}`);
    return t;
  };
  const byOutput = new Map<string, RawNode>();
  for (const n of nodes) for (const o of n.outputs) byOutput.set(o, n);

  const linear = (output: string, biasName: string): Q4Linear => {
    const node = byOutput.get(output);
    if (!node || node.opType !== "MatMulNBits") {
      throw new Error(`onnx: ${output} is not a MatMulNBits (webgpu2 needs the q4 export)`);
    }
    if (node.inputs.length > 3 && node.inputs[3]) throw new Error(`onnx: ${output} has zero points; unsupported`);
    const q = get(node.inputs[1]);
    const [n, blocks, blockBytes] = q.dims;
    if (blockBytes !== 16) throw new Error(`onnx: ${output} block_size ${blockBytes * 2}, expected 32`);
    if (!q.raw) throw new Error(`onnx: ${output} weights not in raw_data`);
    return { n, k: blocks * 32, q: toU32Words(q.raw), scales: toFloat32(get(node.inputs[2])), bias: toFloat32(get(biasName)) };
  };

  const table = (prefix: string): U8Table => {
    const q = get(`${prefix}.weight_quantized`);
    if (!q.raw) throw new Error(`onnx: ${prefix} not in raw_data`);
    return {
      rows: q.dims[0],
      cols: q.dims[1],
      q: toU32Words(q.raw),
      scale: scalar(get(`${prefix}.weight_scale`)),
      zeroPoint: scalar(get(`${prefix}.weight_zero_point`)),
    };
  };
  const ln = (prefix: string) => ({ gamma: toFloat32(get(`${prefix}.weight`)), beta: toFloat32(get(`${prefix}.bias`)) });

  const layers: BertLayer[] = [];
  for (let l = 0; l < config.num_hidden_layers; l++) {
    const p = `/bert/encoder/layer.${l}`;
    const w = `bert.encoder.layer.${l}`;
    const q = linear(`${p}/attention/self/query/MatMul_output_0`, `${w}.attention.self.query.bias`);
    const k = linear(`${p}/attention/self/key/MatMul_output_0`, `${w}.attention.self.key.bias`);
    const v = linear(`${p}/attention/self/value/MatMul_output_0`, `${w}.attention.self.value.bias`);
    layers.push({
      qkv: concatLinears([q, k, v]),
      attnOut: linear(`${p}/attention/output/dense/MatMul_output_0`, `${w}.attention.output.dense.bias`),
      ln1: ln(`${w}.attention.output.LayerNorm`),
      ffnIn: linear(`${p}/intermediate/dense/MatMul_output_0`, `${w}.intermediate.dense.bias`),
      ffnOut: linear(`${p}/output/dense/MatMul_output_0`, `${w}.output.dense.bias`),
      ln2: ln(`${w}.output.LayerNorm`),
    });
  }

  return {
    hidden: config.hidden_size,
    heads: config.num_attention_heads,
    intermediate: config.intermediate_size,
    numLabels: Object.keys(config.id2label).length,
    maxPositions: config.max_position_embeddings,
    layerNormEps: config.layer_norm_eps,
    wordEmb: table("bert.embeddings.word_embeddings"),
    posEmb: table("bert.embeddings.position_embeddings"),
    typeEmb: table("bert.embeddings.token_type_embeddings"),
    embLn: ln("bert.embeddings.LayerNorm"),
    layers,
    classifier: linear("/classifier/MatMul_output_0", "classifier.bias"),
  };
}

/** Stack linears that share K along N. MatMulNBits is N-major, so this is a plain concat. */
function concatLinears(parts: readonly Q4Linear[]): Q4Linear {
  const cat = <T extends Uint32Array | Float32Array>(arrs: T[], make: (n: number) => T): T => {
    const out = make(arrs.reduce((s, a) => s + a.length, 0));
    let o = 0;
    for (const a of arrs) {
      out.set(a, o);
      o += a.length;
    }
    return out;
  };
  return {
    n: parts.reduce((s, p) => s + p.n, 0),
    k: parts[0].k,
    q: cat(parts.map((p) => p.q), (n) => new Uint32Array(n)),
    scales: cat(parts.map((p) => p.scales), (n) => new Float32Array(n)),
    bias: cat(parts.map((p) => p.bias), (n) => new Float32Array(n)),
  };
}

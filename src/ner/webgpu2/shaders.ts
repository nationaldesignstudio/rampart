/**
 * WGSL kernels for the webgpu2 backend: the shipped q4 BERT token classifier
 * (6 layers, hidden 384, 12 heads of 32, FFN 1536) as 44 dispatches in one
 * compute pass, one submit, one readback.
 *
 * What the ORT profile said, and what each kernel does about it:
 *
 *  - The q4 MatMul was 71% of GPU time. Here the 4-bit blocks stay packed in
 *    VRAM exactly as MatMulNBits stores them (16 bytes = 32 weights + one fp32
 *    scale) and are unpacked in registers/shared memory inside the GEMM tile,
 *    so there is no separate dequant pass and no fp32 weight copy.
 *  - The ~5.5 ms floor at short lengths was ORT's per-op overhead (~300 graph
 *    nodes → dozens of dispatches, plus shape ops and per-run allocation). Here
 *    every buffer is preallocated at 512 positions, bind groups are built once,
 *    and a layer is seven kernels: QKV (one fused N=1152 GEMM), attention
 *    (online softmax, no score matrix in memory), out-proj + bias + residual,
 *    LayerNorm, FFN-in + bias + GELU, FFN-out + bias + residual, LayerNorm.
 *    Embedding gather+LN and classifier+softmax+argmax are one kernel each.
 *
 * LayerNorm is deliberately *not* fused into the GEMMs that feed it. A fused
 * kernel has to own whole 384-wide rows, which at T≈30 leaves 4 workgroups
 * walking K=1536 serially; measured, that was ~90% of the runtime. The tiled
 * GEMM spreads the same work over 6–24× more workgroups, and the separate LN
 * dispatch costs a few microseconds.
 *
 * Kernels are generated per shape (K/N baked in as constants) rather than
 * reading them from uniforms, so every loop bound is static. The only runtime
 * value is the token count, and it only guards loads and stores.
 */

export interface ShaderDims {
  readonly hidden: number;
  readonly heads: number;
  readonly intermediate: number;
  readonly numLabels: number;
  readonly vocab: number;
  readonly eps: number;
}

const f = (x: number): string => (Number.isInteger(x) ? `${x}.0` : `${x}`);

const PARAMS = /* wgsl */ `
struct Params { tokens: u32, _p0: u32, _p1: u32, _p2: u32 };
`;

/** Workgroup tree-sum over WG lanes. Must be called from uniform control flow. */
const wgSum = (lanes: number): string => /* wgsl */ `
var<workgroup> red: array<f32, ${lanes}>;
fn wg_sum(x: f32, t: u32) -> f32 {
  red[t] = x;
  workgroupBarrier();
  for (var s = ${lanes / 2}u; s > 0u; s = s >> 1u) {
    if (t < s) { red[t] = red[t] + red[t + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
`;

/**
 * Embedding gather (three per-tensor uint8 tables, dequantized inline) +
 * LayerNorm. One workgroup per token.
 */
export function embedShader(
  d: ShaderDims,
  q: { word: [number, number]; pos: [number, number]; type: [number, number] },
): string {
  const lanes = 128;
  const per = d.hidden / lanes;
  return /* wgsl */ `
${PARAMS}
const H: u32 = ${d.hidden}u;
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> word: array<u32>;
@group(0) @binding(2) var<storage, read> pos: array<u32>;
@group(0) @binding(3) var<storage, read> typ: array<u32>;
@group(0) @binding(4) var<storage, read> ln: array<f32>;   // gamma[H] | beta[H]
@group(0) @binding(5) var<storage, read_write> h: array<f32>;
${wgSum(lanes)}
fn deq(w: u32, c: u32, scale: f32, zp: f32) -> f32 {
  return (f32((w >> (8u * (c & 3u))) & 255u) - zp) * scale;
}
@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let row = wg.x;
  let id = min(ids[row], ${d.vocab - 1}u);
  var v: array<f32, ${per}>;
  var s = 0.0;
  for (var j = 0u; j < ${per}u; j++) {
    let c = t + ${lanes}u * j;
    // (word + token_type) + position: the graph's Add order.
    let x = deq(word[(id * H + c) >> 2u], c, ${f(q.word[0])}, ${f(q.word[1])})
          + deq(typ[c >> 2u], c, ${f(q.type[0])}, ${f(q.type[1])})
          + deq(pos[(row * H + c) >> 2u], c, ${f(q.pos[0])}, ${f(q.pos[1])});
    v[j] = x;
    s += x;
  }
  let mean = wg_sum(s, t) / f32(H);
  var s2 = 0.0;
  for (var j = 0u; j < ${per}u; j++) { let dv = v[j] - mean; s2 += dv * dv; }
  let inv = inverseSqrt(wg_sum(s2, t) / f32(H) + ${d.eps});
  for (var j = 0u; j < ${per}u; j++) {
    let c = t + ${lanes}u * j;
    h[row * H + c] = (v[j] - mean) * inv * ln[c] + ln[H + c];
  }
}
`;
}

// Abramowitz & Stegun 7.1.26: |err| < 1.5e-7, below fp32 GEMM noise.
const ERF = /* wgsl */ `
fn erf_(x: f32) -> f32 {
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return select(-y, y, x >= 0.0);
}
`;

/**
 * Tiled q4 GEMM, out[T, N] = A[T, K] · Wᵀ + bias, then optionally GELU, or
 * optionally + res[T, N] (the residual stream, added before LayerNorm).
 *
 * 256 lanes compute a 32×64 output tile, 2×4 per lane. Each 32-deep K step
 * stages A's 32×32 slab and the 64 columns' q4 blocks — 64 × 16 bytes, one
 * u32 (8 nibbles) per lane — dequantized once into shared memory, so every
 * weight is unpacked exactly once per tile and read 32× from shared.
 *
 * At chat-sized T this kernel is latency-bound, not FLOP-bound: a handful of
 * workgroups each walk K serially, one global round trip per step. Two things
 * shorten that chain:
 *  - the next step's A and W are fetched into registers before the current
 *    step's FMAs run, so the load latency overlaps compute;
 *  - `splitK` > 1 gives each workgroup (grid z) a 1/splitK slice of K and
 *    writes partial sums to `out[z]`, [splitK][maxT][N]. Only slice 0 adds
 *    bias/residual; the LayerNorm kernel that consumes these sums the slices.
 *    Used only for FFN-out (K=1536), the one place it measurably paid.
 *
 * Measured and rejected: 64-deep K steps, 16-row tiles, a vec4/4-accumulator
 * ILP variant, and split-K on the K=384 GEMMs — none beat this on M2 Max, and
 * the latter two launch 2–4× the workgroups, which costs more on mobile GPUs.
 * Shared memory is 12.4 KiB and every kernel fits WebGPU's default limits, so
 * no optional features or raised limits are requested.
 */
export function gemmShader(
  k: number,
  n: number,
  epilogue: { readonly gelu?: boolean; readonly residual?: boolean; readonly splitK?: number; readonly maxTokens?: number } = {},
): string {
  const { gelu = false, residual = false, splitK = 1, maxTokens = 512 } = epilogue;
  const TM = 32;
  const TN = 64;
  if (n % TN !== 0 || k % (32 * splitK) !== 0) throw new Error(`gemm: unsupported shape K=${k} N=${n} splitK=${splitK}`);
  if (splitK > 1 && gelu) throw new Error("gemm: GELU needs the full K sum; it can't split");
  return /* wgsl */ `
${PARAMS}
const K: u32 = ${k}u;
const N: u32 = ${n}u;
const NB: u32 = ${k / 32}u;
const NB_SLICE: u32 = ${k / 32 / splitK}u;
@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> S: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
${residual ? "@group(0) @binding(6) var<storage, read> res: array<f32>;" : ""}
var<workgroup> As: array<f32, ${TM * 33}>;   // [row][k], padded against bank conflicts
var<workgroup> Ws: array<f32, ${32 * TN}>;   // [k][col]
${gelu ? ERF : ""}
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let T = params.tokens;
  let row0 = wg.y * ${TM}u;
  let col0 = wg.x * ${TN}u;
  let slice = wg.z;
  let tx = lid.x;
  let ty = lid.y;
  var acc: array<f32, 8>;

  // Staging roles: A rows 0..31 × 8 vec4 per row; W 64 cols × 4 words.
  let ar = li >> 3u;
  let ac = (li & 7u) * 4u;
  let arow = row0 + ar;
  let aok = arow < T;
  let abase = (arow * K + ac) >> 2u;
  let wcol = li >> 2u;
  let wi = li & 3u;
  let wn = col0 + wcol;
  let kb0 = slice * NB_SLICE;

  var ra = select(vec4<f32>(0.0), A[abase + kb0 * 8u], aok);
  var rw = W[(wn * NB + kb0) * 4u + wi];
  var rs = S[wn * NB + kb0];

  for (var step = 0u; step < NB_SLICE; step++) {
    As[ar * 33u + ac] = ra.x;
    As[ar * 33u + ac + 1u] = ra.y;
    As[ar * 33u + ac + 2u] = ra.z;
    As[ar * 33u + ac + 3u] = ra.w;
    for (var j = 0u; j < 8u; j++) {
      Ws[(wi * 8u + j) * ${TN}u + wcol] = (f32((rw >> (4u * j)) & 15u) - 8.0) * rs;
    }
    workgroupBarrier();
    // Issue the next step's loads now; they land while the FMAs below run.
    if (step + 1u < NB_SLICE) {
      let kb = kb0 + step + 1u;
      ra = select(vec4<f32>(0.0), A[abase + kb * 8u], aok);
      rw = W[(wn * NB + kb) * 4u + wi];
      rs = S[wn * NB + kb];
    }
    for (var kk = 0u; kk < 32u; kk++) {
      let a0 = As[ty * 33u + kk];
      let a1 = As[(ty + 16u) * 33u + kk];
      for (var j = 0u; j < 4u; j++) {
        let w = Ws[kk * ${TN}u + tx + 16u * j];
        acc[j] = fma(a0, w, acc[j]);
        acc[4u + j] = fma(a1, w, acc[4u + j]);
      }
    }
    workgroupBarrier();
  }

  let first = slice == 0u;
  for (var i = 0u; i < 2u; i++) {
    let row = row0 + ty + 16u * i;
    if (row >= T) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let col = col0 + tx + 16u * j;
      var v = acc[i * 4u + j] + select(0.0, bias[col], first);
      ${gelu ? "v = 0.5 * v * (1.0 + erf_(v * 0.7071067811865476));" : ""}
      ${residual ? "if (first) { v += res[row * N + col]; }" : ""}
      out[(slice * ${maxTokens}u + row) * N + col] = v;
    }
  }
}
`;
}

/**
 * Row LayerNorm, h = LN(Σ x[slice]) · gamma + beta. One workgroup per token;
 * two-pass (mean, then centered variance) like ORT's kernel. Summing the
 * split-K partials here is what lets the GEMM before it split K for free.
 */
export function layerNormShader(d: ShaderDims, slices = 1, maxTokens = 512): string {
  const lanes = 128;
  const per = d.hidden / lanes;
  return /* wgsl */ `
const H: u32 = ${d.hidden}u;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> ln: array<f32>;   // gamma[H] | beta[H]
@group(0) @binding(2) var<storage, read_write> h: array<f32>;
${wgSum(lanes)}
@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let base = wg.x * H;
  var v: array<f32, ${per}>;
  var s = 0.0;
  for (var j = 0u; j < ${per}u; j++) {
    var xv = 0.0;
    // Sum split-K partials, [slices][maxT][H].
    for (var p = 0u; p < ${slices}u; p++) { xv += x[p * ${maxTokens * d.hidden}u + base + t + ${lanes}u * j]; }
    v[j] = xv;
    s += xv;
  }
  let mean = wg_sum(s, t) / f32(H);
  var s2 = 0.0;
  for (var j = 0u; j < ${per}u; j++) { let dv = v[j] - mean; s2 += dv * dv; }
  let inv = inverseSqrt(wg_sum(s2, t) / f32(H) + ${d.eps});
  for (var j = 0u; j < ${per}u; j++) {
    let c = t + ${lanes}u * j;
    h[base + c] = (v[j] - mean) * inv * ln[c] + ln[H + c];
  }
}
`;
}

/**
 * Self-attention for one head over a 16-query block, streaming keys in chunks
 * of 32 with an online softmax, so the T×T score matrix never exists and the
 * kernel works for any T up to the position limit. 128 lanes = 16 queries × 8
 * lanes; a lane scores 4 keys per chunk and owns 4 of the head's 32 output
 * dims. The attention mask is all ones for a single unpadded sequence, so
 * there is nothing to mask beyond the T bound.
 */
export function attentionShader(d: ShaderDims): string {
  const H = d.hidden;
  const HD = H / d.heads;
  if (HD !== 32) throw new Error(`attention: head dim ${HD} unsupported (expects 32)`);
  return /* wgsl */ `
${PARAMS}
const H: u32 = ${H}u;
const QKV: u32 = ${3 * H}u;
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> ctx: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> Qs: array<f32, 512>;    // [16 q][32 d]
var<workgroup> Ks: array<f32, ${32 * 33}>;   // [32 key][33], padded
var<workgroup> Vs: array<f32, 1024>;   // [32 key][32 d]
var<workgroup> Ss: array<f32, 512>;    // [16 q][32 key]

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let T = params.tokens;
  let head = wg.y;
  let q0 = wg.x * 16u;
  let qi = li >> 3u;
  let sub = li & 7u;
  let hoff = head * 32u;
  let scale = ${f(1 / Math.sqrt(HD))};

  for (var i = 0u; i < 4u; i++) {
    let idx = li + 128u * i;
    let row = q0 + (idx >> 5u);
    Qs[idx] = select(0.0, qkv[row * QKV + hoff + (idx & 31u)] * scale, row < T);
  }

  var m = -3.0e38;
  var l = 0.0;
  var acc = vec4<f32>(0.0);
  for (var k0 = 0u; k0 < T; k0 += 32u) {
    workgroupBarrier();
    for (var i = 0u; i < 8u; i++) {
      let idx = li + 128u * i;
      let r = idx >> 5u;
      let dd = idx & 31u;
      let key = k0 + r;
      let ok = key < T;
      // Clamp so the tail chunk never indexes past the buffer; masked below.
      let kr = min(key, T - 1u) * QKV;
      Ks[r * 33u + dd] = select(0.0, qkv[kr + H + hoff + dd], ok);
      Vs[idx] = select(0.0, qkv[kr + 2u * H + hoff + dd], ok);
    }
    workgroupBarrier();
    for (var j = 0u; j < 4u; j++) {
      let kk = sub + 8u * j;
      var s = 0.0;
      for (var dd = 0u; dd < 32u; dd++) { s = fma(Qs[qi * 32u + dd], Ks[kk * 33u + dd], s); }
      Ss[qi * 32u + kk] = select(-3.0e38, s, k0 + kk < T);
    }
    workgroupBarrier();
    var cm = -3.0e38;
    for (var kk = 0u; kk < 32u; kk++) { cm = max(cm, Ss[qi * 32u + kk]); }
    let nm = max(m, cm);
    let corr = exp(m - nm);
    l *= corr;
    acc *= corr;
    for (var kk = 0u; kk < 32u; kk++) {
      let p = exp(Ss[qi * 32u + kk] - nm);
      l += p;
      let vb = kk * 32u + sub * 4u;
      acc += p * vec4<f32>(Vs[vb], Vs[vb + 1u], Vs[vb + 2u], Vs[vb + 3u]);
    }
    m = nm;
  }

  let row = q0 + qi;
  if (row < T) {
    let o = row * H + hoff + sub * 4u;
    let r = acc / l;
    ctx[o] = r.x;
    ctx[o + 1u] = r.y;
    ctx[o + 2u] = r.z;
    ctx[o + 3u] = r.w;
  }
}
`;
}

/**
 * Classifier head + softmax + argmax, one workgroup per token. Writes the
 * winning label and its softmax probability — exactly what the
 * transformers.js token-classification pipeline reports — plus the raw
 * logits for parity testing.
 */
export function classifierShader(d: ShaderDims): string {
  const H = d.hidden;
  const L = d.numLabels;
  const lanes = 64;
  if (L > lanes) throw new Error(`classifier: ${L} labels exceeds ${lanes} lanes`);
  return /* wgsl */ `
${PARAMS}
const H: u32 = ${H}u;
const L: u32 = ${L}u;
const NB: u32 = ${H / 32}u;
@group(0) @binding(0) var<storage, read> h: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> S: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> result: array<u32>;   // [T][2]: label, score bits
@group(0) @binding(5) var<storage, read_write> logits: array<f32>;   // [T][L]
var<workgroup> x: array<f32, ${H}>;
var<workgroup> lg: array<f32, ${lanes}>;

fn nib_dot(w: u32, base: u32) -> f32 {
  var s = 0.0;
  for (var b = 0u; b < 8u; b++) { s = fma(f32((w >> (4u * b)) & 15u) - 8.0, x[base + b], s); }
  return s;
}

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let row = wg.x;
  for (var c = t; c < H; c += ${lanes}u) { x[c] = h[row * H + c]; }
  workgroupBarrier();
  if (t < L) {
    var acc = 0.0;
    for (var kb = 0u; kb < NB; kb++) {
      let w = W[t * NB + kb];
      let base = kb * 32u;
      acc = fma(S[t * NB + kb], nib_dot(w.x, base) + nib_dot(w.y, base + 8u) + nib_dot(w.z, base + 16u) + nib_dot(w.w, base + 24u), acc);
    }
    let v = acc + bias[t];
    lg[t] = v;
    logits[row * L + t] = v;
  }
  workgroupBarrier();
  if (t == 0u) {
    var best = 0u;
    var mx = lg[0];
    for (var i = 1u; i < L; i++) { if (lg[i] > mx) { mx = lg[i]; best = i; } }
    var sum = 0.0;
    for (var i = 0u; i < L; i++) { sum += exp(lg[i] - mx); }
    result[row * 2u] = best;
    result[row * 2u + 1u] = bitcast<u32>(1.0 / sum);
  }
}
`;
}

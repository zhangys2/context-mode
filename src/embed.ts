/**
 * embed — Local, offline sentence embeddings for the vector search layer.
 *
 * DEVIATION FROM THE ORIGINAL "WASM-only" DESIGN — see PR description for
 * full detail. The design called for a pure-WASM backend so this stays
 * consistent with db-base.ts's deliberate avoidance of per-platform native
 * binaries (bun:sqlite/node:sqlite over better-sqlite3 wherever possible).
 * That turned out not to be achievable with @huggingface/transformers under
 * plain Node.js:
 *   - The package's Node build (resolved by Node's "node" export condition,
 *     which always wins at runtime here) only accepts device "cpu" | "dml" |
 *     "webgpu" — there is no "wasm" device, and "cpu" runs on
 *     onnxruntime-node's NATIVE addon, not WASM.
 *   - Forcing the package's browser build (dist/transformers.web.js) via a
 *     direct file-path import does expose device "cpu" → onnxruntime-web
 *     (real WASM), but its model/tokenizer file loading assumes a browser
 *     environment and fails under bare Node with "Unable to get model file
 *     path or buffer" after download completes — not a viable path without
 *     reimplementing file loading ourselves.
 * So this uses the Node build's onnxruntime-node CPU backend, which DOES
 * reintroduce a per-platform native addon dependency, scoped only to this
 * embedding pipeline (SQLite driver selection in db-base.ts is unaffected).
 * Never throws regardless: callers treat a null return as "vector layer
 * unavailable for this call" and fall back to lexical-only search/indexing.
 */

const DEBUG = process.env.DEBUG?.includes("context-mode");

const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
export const EMBEDDING_DIM = 384;

type FeatureExtractionPipeline = (
  text: string,
  options: { pooling: "mean"; normalize: boolean },
) => Promise<{ data: Float32Array | number[] }>;

let _pipelinePromise: Promise<FeatureExtractionPipeline | null> | null = null;
let _warned = false;

function warnOnce(context: string, err: unknown): void {
  if (_warned) return;
  _warned = true;
  if (DEBUG) {
    const msg = err instanceof Error ? err.message : String(err);
    process.stderr.write(`[ctx] embedding unavailable (${context}): ${msg}\n`);
  }
}

async function loadPipeline(): Promise<FeatureExtractionPipeline | null> {
  if (!_pipelinePromise) {
    _pipelinePromise = (async () => {
      try {
        const { pipeline } = await import("@huggingface/transformers");
        // "cpu" here is onnxruntime-node's native CPU execution provider —
        // see the module doc comment above for why this isn't WASM.
        // dtype: "q8" selects the ~23MB quantized weights (not the ~90MB
        // fp32 default), matching the original size target.
        const extractor = await pipeline("feature-extraction", MODEL_ID, {
          device: "cpu",
          dtype: "q8",
        });
        return extractor as unknown as FeatureExtractionPipeline;
      } catch (err) {
        warnOnce("pipeline load", err);
        return null;
      }
    })();
  }
  return _pipelinePromise;
}

/**
 * Embed a chunk of text into a 384-dim normalized vector.
 * Returns null on any failure (model load, WASM unsupported, inference
 * error, empty input) — callers must degrade to lexical-only, never throw.
 */
export async function embed(text: string): Promise<Float32Array | null> {
  if (!text || text.trim().length === 0) return null;
  try {
    const extractor = await loadPipeline();
    if (!extractor) return null;
    const output = await extractor(text, { pooling: "mean", normalize: true });
    return output.data instanceof Float32Array ? output.data : new Float32Array(output.data);
  } catch (err) {
    warnOnce("inference", err);
    return null;
  }
}

/** Cosine similarity between two equal-length vectors. Assumes normalized input but doesn't require it. */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** Pack a Float32Array into a Buffer for BLOB storage. */
export function encodeVector(v: Float32Array): Buffer {
  return Buffer.from(v.buffer, v.byteOffset, v.byteLength);
}

/** Unpack a BLOB column back into a Float32Array. Copies the bytes so the returned array is safe to keep past the row's lifetime. */
export function decodeVector(buf: Buffer): Float32Array {
  const copy = Uint8Array.from(buf);
  return new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);
}

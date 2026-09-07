/**
 * embed — local sentence-embedding pipeline tests.
 *
 * The real `embed()` call downloads/loads an ONNX model on first use
 * (cached afterward) — slow and network-dependent, hence the extended
 * per-test timeout below. cosineSimilarity/encodeVector/decodeVector are
 * pure functions and need no model.
 */

import { describe, test, expect } from "vitest";
import { embed, cosineSimilarity, encodeVector, decodeVector } from "../src/embed.js";

const MODEL_TIMEOUT = 60_000;

describe("cosineSimilarity", () => {
  test("identical vectors → similarity ≈ 1", () => {
    const v = new Float32Array([0.6, 0.8, 0]);
    expect(cosineSimilarity(v, v)).toBeCloseTo(1, 5);
  });

  test("orthogonal vectors → similarity ≈ 0", () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([0, 1]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(0, 5);
  });

  test("opposite vectors → similarity ≈ -1", () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([-1, 0]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1, 5);
  });

  test("zero vector → similarity 0 (no NaN/divide-by-zero)", () => {
    const a = new Float32Array([0, 0, 0]);
    const b = new Float32Array([1, 2, 3]);
    expect(cosineSimilarity(a, b)).toBe(0);
  });
});

describe("encodeVector / decodeVector", () => {
  test("round-trips a Float32Array exactly through a Buffer", () => {
    const original = new Float32Array([1.5, -2.25, 0, 3.140000104904175, -0.0001]);
    const buf = encodeVector(original);
    expect(buf).toBeInstanceOf(Buffer);
    const decoded = decodeVector(buf);
    expect(decoded.length).toBe(original.length);
    for (let i = 0; i < original.length; i++) {
      expect(decoded[i]).toBe(original[i]);
    }
  });

  test("decoded vector is independent of the source buffer's later mutation", () => {
    const original = new Float32Array([1, 2, 3]);
    const buf = encodeVector(original);
    const decoded = decodeVector(buf);
    buf.fill(0);
    expect(decoded[0]).toBe(1);
  });
});

describe("embed", () => {
  test("returns null for empty/whitespace input", async () => {
    expect(await embed("")).toBeNull();
    expect(await embed("   ")).toBeNull();
  });

  test(
    "embeds real text into a 384-dim vector, usable for cosine similarity",
    async () => {
      const v1 = await embed("The cat sat on the mat.");
      const v2 = await embed("A feline rested on the rug.");
      const v3 = await embed("Quarterly revenue increased by twelve percent.");

      expect(v1).not.toBeNull();
      expect(v2).not.toBeNull();
      expect(v3).not.toBeNull();
      expect(v1!.length).toBe(384);

      const simRelated = cosineSimilarity(v1!, v2!);
      const simUnrelated = cosineSimilarity(v1!, v3!);
      // Semantically related sentences (no shared vocabulary) should score
      // higher than an unrelated one — the whole point of the vector layer.
      expect(simRelated).toBeGreaterThan(simUnrelated);
    },
    MODEL_TIMEOUT,
  );
});

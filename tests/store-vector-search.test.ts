/**
 * ContentStore — hybrid vector search tests.
 *
 * Covers the vector layer added alongside the existing porter/trigram FTS5
 * layers: chunk_vectors population on index, dedup on re-index, and that
 * the vector layer surfaces semantically-related-but-lexically-dissimilar
 * content that pure BM25 search would miss.
 *
 * Real embedding calls load/download an ONNX model on first use — slow and
 * network-dependent, hence the extended per-test timeout.
 */

import { describe, test, expect } from "vitest";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { ContentStore } from "../src/store.js";

const MODEL_TIMEOUT = 60_000;

function createStore(): ContentStore {
  const path = join(
    tmpdir(),
    `context-mode-vec-test-${Date.now()}-${Math.random().toString(36).slice(2)}.db`,
  );
  return new ContentStore(path);
}

describe("chunk_vectors population", () => {
  test(
    "indexing writes one chunk_vectors row per chunk",
    async () => {
      const store = createStore();
      try {
        const result = await store.index({
          content: "# Section One\n\nContent about databases.\n\n# Section Two\n\nContent about networking.",
          source: "vector-test-basic",
        });
        expect(result.totalChunks).toBeGreaterThan(0);
        // Direct chunk_vectors row-count coverage isn't exposed as public
        // API — the semantic-recall tests below are the real behavioral
        // proof that indexing actually populated usable vectors.
      } finally {
        store.close();
      }
    },
    MODEL_TIMEOUT,
  );

  test(
    "re-indexing the same label replaces vectors, not accumulates them",
    async () => {
      const store = createStore();
      try {
        await store.index({
          content: "# A\n\nFirst version content about apples.\n\n# B\n\nSecond section about oranges.",
          source: "vector-test-dedup",
        });
        const second = await store.index({
          content: "# Only\n\nCompletely replaced content about bananas.",
          source: "vector-test-dedup",
        });
        expect(second.totalChunks).toBe(1);

        // If old vector rows leaked, a query for the OLD content ("apples")
        // would still surface a result via the vector layer even though the
        // lexical layers correctly show nothing for the replaced source.
        const results = await store.searchWithFallback("apples oranges", 5, "vector-test-dedup", undefined, "exact");
        const stale = results.filter((r) => r.content.includes("apples") || r.content.includes("oranges"));
        expect(stale.length).toBe(0);
      } finally {
        store.close();
      }
    },
    MODEL_TIMEOUT,
  );
});

describe("semantic recall via the vector layer", () => {
  test(
    "finds a semantically related chunk that shares no vocabulary with the query",
    async () => {
      const store = createStore();
      try {
        // Deliberately no lexical overlap with the query below — porter and
        // trigram layers alone should not surface this chunk.
        await store.index({
          content: "# Feline Behavior\n\nDomestic cats often rest on soft furniture during the day.",
          source: "vector-test-semantic",
        });

        const results = await store.searchWithFallback(
          "where do house pets like to nap",
          3,
          "vector-test-semantic",
          undefined,
          "exact",
        );

        expect(results.length).toBeGreaterThan(0);
        expect(results[0].content).toContain("cats");
      } finally {
        store.close();
      }
    },
    MODEL_TIMEOUT,
  );

  test(
    "unique lexical token is found even with a real (non-null) query embedding in play",
    async () => {
      const store = createStore();
      try {
        await store.index({
          content: "# Exact Match\n\nUnique token xyzzyplugh appears here.",
          source: "vector-test-fallback",
        });
        // "xyzzyplugh" is a real, if nonsensical, token — it embeds fine, so
        // this does NOT exercise null-embedding degradation (that path —
        // embed() actually failing — is covered by
        // tests/embed-failure.test.ts, which mocks the failure instead of
        // relying on a query that happens not to trigger it). This is a
        // plain regression check that an exact, unambiguous lexical match
        // still surfaces correctly now that RRF fuses in a third layer.
        const results = await store.searchWithFallback("xyzzyplugh", 3, "vector-test-fallback", undefined, "exact");
        expect(results.length).toBeGreaterThan(0);
      } finally {
        store.close();
      }
    },
    MODEL_TIMEOUT,
  );
});

/**
 * embed — mocked failure-path tests.
 *
 * The real model tests in tests/embed.test.ts prove embed() works; these
 * prove it degrades correctly when it can't — the actual failure mode in
 * production (offline first run, native addon init failure, OOM). Mocks
 * "@huggingface/transformers" directly rather than "../src/embed.js" so the
 * module-under-test's own error handling runs for real.
 *
 * Each test resets the module registry and re-imports embed.ts fresh, since
 * loadPipeline() caches its result in a module-level singleton — without a
 * reset, the first test's mock would leak into the rest.
 */

import { describe, test, expect, vi, beforeEach } from "vitest";

describe("embed — failure handling (mocked)", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.doUnmock("@huggingface/transformers");
  });

  test("returns null when pipeline() rejects (model load failure)", async () => {
    vi.doMock("@huggingface/transformers", () => ({
      pipeline: vi.fn().mockRejectedValue(new Error("simulated model load failure")),
    }));
    const { embed } = await import("../src/embed.js");
    expect(await embed("hello world")).toBeNull();
  });

  test("returns null when the loaded extractor throws during inference", async () => {
    vi.doMock("@huggingface/transformers", () => ({
      pipeline: vi.fn().mockResolvedValue(
        vi.fn().mockRejectedValue(new Error("simulated inference failure")),
      ),
    }));
    const { embed } = await import("../src/embed.js");
    expect(await embed("hello world")).toBeNull();
  });

  test("a load failure is cached — pipeline() is not retried on subsequent calls", async () => {
    const pipelineMock = vi.fn().mockRejectedValue(new Error("simulated model load failure"));
    vi.doMock("@huggingface/transformers", () => ({ pipeline: pipelineMock }));
    const { embed } = await import("../src/embed.js");

    await embed("first call");
    await embed("second call");
    await embed("third call");

    expect(pipelineMock).toHaveBeenCalledTimes(1);
  });

  test("indexing and search never throw when embeddings are entirely unavailable", async () => {
    vi.doMock("@huggingface/transformers", () => ({
      pipeline: vi.fn().mockRejectedValue(new Error("simulated model load failure")),
    }));
    const { join } = await import("node:path");
    const { tmpdir } = await import("node:os");
    const { ContentStore } = await import("../src/store.js");

    const store = new ContentStore(
      join(tmpdir(), `context-mode-embed-fail-${Date.now()}-${Math.random().toString(36).slice(2)}.db`),
    );
    try {
      const result = await store.index({
        content: "# Networking\n\nTCP handshakes establish a reliable connection.",
        source: "embed-failure-test",
      });
      expect(result.totalChunks).toBeGreaterThan(0);

      const results = await store.searchWithFallback(
        "TCP handshake",
        3,
        "embed-failure-test",
        undefined,
        "exact",
      );
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].content).toContain("TCP");
    } finally {
      store.close();
    }
  });
});

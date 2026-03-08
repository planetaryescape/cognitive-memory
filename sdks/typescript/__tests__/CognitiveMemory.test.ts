import { InMemoryAdapter } from "../src/adapters/memory";
import { CognitiveMemory } from "../src/core/CognitiveMemory";
import type {
  EmbeddingProvider,
  Memory,
  MemoryCategory,
} from "../src/core/types";
import { createDefaultMemory } from "../src/core/types";

function providerFromMap(map: Map<string, number[]>): EmbeddingProvider {
  return {
    async embed(text: string) {
      const v = map.get(text);
      if (!v) throw new Error(`missing embedding for: ${text}`);
      return v;
    },
  };
}

describe("CognitiveMemory", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-02-10T00:00:00.000Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test("store() applies defaults", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([["a", [1, 0]]]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
    });

    const id = await memory.store({ content: "a" });
    const m = await adapter.getMemory(id);
    expect(m?.category).toBe("semantic");
    expect(m?.importance).toBe(0.5);
    expect(m?.stability).toBe(0.3);
    expect(m?.accessCount).toBe(0);
    expect(m?.retention).toBe(1.0);
  });

  test("retrieve() scores by relevance * retention and strengthens memories + links", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([
      ["q", [1, 0]],
      ["A", [1, 0]],
      ["B", [1, 0]],
      ["C", [0, 1]],
    ]);

    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
    });

    const now = Date.now();
    const aId = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "A",
        embedding: embeddings.get("A")!,
      }),
      category: "episodic" as MemoryCategory,
      importance: 0.5,
      stability: 0.5,
      accessCount: 0,
      lastAccessed: now - 1 * 24 * 60 * 60 * 1000,
      retention: 1,
    });
    const bId = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "B",
        embedding: embeddings.get("B")!,
      }),
      category: "episodic" as MemoryCategory,
      importance: 0.5,
      stability: 0.5,
      accessCount: 0,
      lastAccessed: now - 30 * 24 * 60 * 60 * 1000,
      retention: 1,
    });
    const cId = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "C",
        embedding: embeddings.get("C")!,
      }),
      category: "semantic" as MemoryCategory,
      importance: 0.5,
      stability: 0.5,
      accessCount: 0,
      lastAccessed: now - 1 * 24 * 60 * 60 * 1000,
      retention: 1,
    });

    await adapter.createOrStrengthenLink(aId, cId, 0.4);

    const results = await memory.retrieve({
      query: "q",
      limit: 3,
      includeAssociations: true,
    });
    expect(results[0].id).toBe(aId);
    expect(results.some((r) => r.id === cId)).toBe(true);

    const a = await adapter.getMemory(aId);
    expect(a?.accessCount).toBe(1);
    expect(a?.stability).toBeGreaterThan(0.5);
  });

  test("get() strengthens a memory", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([["x", [1, 0]]]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
    });
    const id = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "x",
        embedding: embeddings.get("x")!,
      }),
      importance: 0.5,
      stability: 0.5,
      accessCount: 0,
      lastAccessed: Date.now() - 10_000,
      retention: 1,
    });
    await memory.get(id);
    const m = await adapter.getMemory(id);
    expect(m?.accessCount).toBe(1);
  });

  test("queryMemories() strengthens returned memories", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([["x", [1, 0]]]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
    });
    const id = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "x",
        embedding: embeddings.get("x")!,
      }),
      importance: 0.5,
      stability: 0.5,
      accessCount: 0,
      lastAccessed: Date.now() - 10_000,
      retention: 1,
    });

    await memory.queryMemories({ limit: 10 });
    const m = await adapter.getMemory(id);
    expect(m?.accessCount).toBe(1);
  });

  test("update() regenerates embedding", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([
      ["old", [1, 0]],
      ["new", [0, 1]],
    ]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
    });
    const id = await memory.store({ content: "old" });
    await memory.update(id, "new");
    const m = await adapter.getMemory(id);
    expect(m?.embedding).toEqual([0, 1]);
  });

  test("consolidate() compresses groups and deletes stale", async () => {
    const adapter = new InMemoryAdapter();
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: { embed: async () => [1, 0] },
      userId: "u1",
    });

    const now = Date.now();
    for (const c of [
      "coffee a",
      "coffee b",
      "coffee c",
      "coffee d",
      "coffee e",
    ]) {
      await adapter.createMemory({
        ...createDefaultMemory({
          id: "tmp",
          userId: "u1",
          content: c,
          embedding: [1, 0],
        }),
        stability: 0.3,
        accessCount: 0,
        lastAccessed: now - 200 * 24 * 60 * 60 * 1000,
        retention: 0.1,
      });
    }

    const staleId = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "s",
        embedding: [1, 0],
      }),
      stability: 0.05,
      importance: 0.1,
      accessCount: 0,
      lastAccessed: now - 200 * 24 * 60 * 60 * 1000,
      retention: 0.01,
    });

    const result = await memory.consolidate();
    expect(result.compressed.length).toBe(1);
    expect(result.deleted).toBe(1);
    expect(await adapter.getMemory(staleId)).toBeNull();
  });

  test("consolidate() refreshes retention before finding fading memories", async () => {
    const adapter = new InMemoryAdapter();
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: { embed: async () => [0, 0] },
      userId: "u1",
    });

    const id = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "x",
        embedding: [1, 0],
      }),
      stability: 0.3,
      accessCount: 0,
      lastAccessed: Date.now() - 150 * 24 * 60 * 60 * 1000,
      retention: 1,
    });

    const result = await memory.consolidate();
    expect(result.decayed.map((d) => d.id)).toContain(id);
  });

  test("link() validates strength", async () => {
    const adapter = new InMemoryAdapter();
    const embeddings = new Map<string, number[]>([["x", [1, 0]]]);
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: providerFromMap(embeddings),
      userId: "u1",
    });
    await expect(memory.link("a", "b", 2)).rejects.toThrow(/Invalid strength/);
  });

  test("store() retries embedding up to 3 attempts", async () => {
    const adapter = new InMemoryAdapter();
    const embed = vi
      .fn<
        Parameters<EmbeddingProvider["embed"]>,
        ReturnType<EmbeddingProvider["embed"]>
      >()
      .mockRejectedValueOnce(new Error("rate limit"))
      .mockRejectedValueOnce(new Error("transient"))
      .mockResolvedValue([1, 0]);

    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: { embed },
      userId: "u1",
    });

    const p = memory.store({ content: "x" });
    await vi.runAllTimersAsync();
    await p;
    expect(embed).toHaveBeenCalledTimes(3);
  });

  test("store() fails after 3 embedding attempts", async () => {
    const adapter = new InMemoryAdapter();
    const embed = vi.fn().mockRejectedValue(new Error("down"));
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: { embed },
      userId: "u1",
    });

    const p = memory.store({ content: "x" });
    const ex = expect(p).rejects.toThrow(/Embedding failed/);
    await vi.runAllTimersAsync();
    await ex;
  });

  test("get() throws on invalid lastAccessed", async () => {
    const adapter = new InMemoryAdapter();
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: { embed: async () => [1, 0] },
      userId: "u1",
    });

    const id = await adapter.createMemory({
      ...createDefaultMemory({
        id: "tmp",
        userId: "u1",
        content: "x",
        embedding: [1, 0],
      }),
      stability: 0.5,
      accessCount: 0,
      lastAccessed: Number.NaN,
      retention: 1,
    });

    await expect(memory.get(id)).rejects.toThrow(/Invalid lastAccessed/);
  });

  test("getStats() returns correct counts", async () => {
    const adapter = new InMemoryAdapter();
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: { embed: async () => [1, 0] },
      userId: "u1",
    });

    await memory.store({ content: "hello", category: "semantic" });
    await memory.store({ content: "world", category: "core" });

    const stats = await memory.getStats();
    expect(stats.total).toBe(2);
    expect(stats.hot).toBe(2);
    expect(stats.cold).toBe(0);
    expect(stats.stub).toBe(0);
    expect(stats.core).toBe(1);
  });

  test("clear() removes all memories", async () => {
    const adapter = new InMemoryAdapter();
    const memory = new CognitiveMemory({
      adapter,
      embeddingProvider: { embed: async () => [1, 0] },
      userId: "u1",
    });

    await memory.store({ content: "hello" });
    await memory.store({ content: "world" });
    await memory.clear();

    const stats = await memory.getStats();
    expect(stats.total).toBe(0);
  });

  describe("extractionMode", () => {
    const fakeLlm = {
      async complete() {
        return JSON.stringify([
          { content: "Ross is a paleontologist", category: "core", importance: 0.9 },
          { content: "Rachel said nice to meet you", category: "episodic", importance: 0.3 },
        ]);
      },
    };

    test("raw mode stores turns verbatim without LLM", async () => {
      const adapter = new InMemoryAdapter();
      const embeddings = new Map<string, number[]>([
        ["Ross: I got you a present. It is a Slinky!", [1, 0]],
        ["Rachel: A Slinky? That is so thoughtful.", [0.9, 0.1]],
        ["Joey: Who wants pizza?", [0, 1]],
      ]);
      const memory = new CognitiveMemory({
        adapter,
        embeddingProvider: providerFromMap(embeddings),
        userId: "u1",
        config: { extractionMode: "raw" },
      });

      const conversation =
        "[This conversation took place on 2024-12-14]\n" +
        "Ross: I got you a present. It is a Slinky!\n" +
        "Rachel: A Slinky? That is so thoughtful.\n" +
        "Joey: Who wants pizza?";

      const ids = await memory.extractAndStore(conversation, "s1", fakeLlm);

      // Should store 3 raw turns (header skipped), no LLM call
      expect(ids).toHaveLength(3);

      const stats = await memory.getStats();
      expect(stats.total).toBe(3);

      // Verify content is verbatim
      const m = await adapter.getMemory(ids[0]);
      expect(m?.content).toContain("Slinky");
      expect(m?.stability).toBe(0.2); // raw turns get stability 0.2
    });

    test("hybrid mode stores both extracted facts and raw turns", async () => {
      const adapter = new InMemoryAdapter();
      const embeddingCalls: string[] = [];
      const memory = new CognitiveMemory({
        adapter,
        embeddingProvider: {
          async embed(text: string) {
            embeddingCalls.push(text);
            return [Math.random(), Math.random()];
          },
        },
        userId: "u1",
        config: { extractionMode: "hybrid" },
      });

      const conversation =
        "Ross: My name is Ross and I am a paleontologist.\n" +
        "Rachel: Nice to meet you Ross!";

      const ids = await memory.extractAndStore(conversation, "s1", fakeLlm);

      // 2 from LLM extraction + 2 raw turns = 4
      expect(ids.length).toBeGreaterThanOrEqual(4);

      const stats = await memory.getStats();
      expect(stats.total).toBeGreaterThanOrEqual(4);
    });

    test("semantic mode (default) does not store raw turns", async () => {
      const adapter = new InMemoryAdapter();
      const memory = new CognitiveMemory({
        adapter,
        embeddingProvider: { embed: async () => [1, 0] },
        userId: "u1",
        // extractionMode defaults to "semantic"
      });

      const conversation = "User: My name is Alice.\nAssistant: Hello Alice!";
      const ids = await memory.extractAndStore(conversation, "s1", fakeLlm);

      // Should only have extracted facts from LLM (2), no raw turns
      expect(ids).toHaveLength(2);

      // Verify none have stability 0.2 (raw turn marker)
      for (const id of ids) {
        const m = await adapter.getMemory(id);
        expect(m?.stability).not.toBe(0.2);
      }
    });

    test("invalid extractionMode throws", async () => {
      const adapter = new InMemoryAdapter();
      const memory = new CognitiveMemory({
        adapter,
        embeddingProvider: { embed: async () => [1, 0] },
        userId: "u1",
        config: { extractionMode: "invalid" as any },
      });

      await expect(
        memory.extractAndStore("User: hello", "s1", fakeLlm),
      ).rejects.toThrow(/Invalid extractionMode/);
    });
  });
});

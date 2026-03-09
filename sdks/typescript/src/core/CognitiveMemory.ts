/**
 * Cognitive Memory System - Main Class
 *
 * High-level API for cognitive memory with decay, retrieval strengthening,
 * associative linking, tiered storage, and LLM extraction.
 */

import type { MemoryAdapter, MemoryFilters } from "../adapters/base";
import { cosineSimilarity } from "../utils/embeddings";
import { extractTopics } from "../utils/scoring";
import { updateStability } from "./decay";
import { CognitiveEngine } from "./engine";
import {
  extractFromConversation,
  extractRawTurns,
  detectConflict,
  compressMemories,
} from "./extraction";
import type { LLMProvider } from "./extraction";
import type {
  CognitiveMemoryConfig,
  ConsolidationResult,
  EmbeddingProvider,
  Memory,
  MemoryInput,
  MemoryStats,
  ResolvedCognitiveMemoryConfig,
  RetrievalQuery,
  ScoredMemory,
  SearchResponse,
  SearchResult,
} from "./types";
import { resolveConfig, getRetentionFloor } from "./types";

function assertNonEmptyString(field: string, value: string) {
  if (value.trim().length === 0) {
    throw new Error(`Invalid ${field}: ${value} (must be non-empty string)`);
  }
}

function assertUnitInterval(field: string, value: number) {
  if (Number.isNaN(value) || value < 0 || value > 1) {
    throw new Error(`Invalid ${field}: ${value} (must be [0.0, 1.0])`);
  }
}

function assertNonNegativeInt(field: string, value: number) {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(
      `Invalid ${field}: ${value} (must be non-negative integer)`,
    );
  }
}

async function sleep(ms: number) {
  await new Promise<void>((resolve) => setTimeout(resolve, ms));
}

/**
 * Main cognitive memory system
 *
 * Provides high-level API for storing, retrieving, and managing memories
 * with human-like characteristics: decay, retrieval strengthening,
 * associative linking, tiered storage, and LLM extraction.
 */
export class CognitiveMemory {
  private adapter: MemoryAdapter;
  private embeddingProvider: EmbeddingProvider;
  private config: ResolvedCognitiveMemoryConfig;
  private engine: CognitiveEngine;

  constructor(options: {
    adapter: MemoryAdapter;
    embeddingProvider: EmbeddingProvider;
    userId: string;
    config?: Partial<CognitiveMemoryConfig>;
  }) {
    assertNonEmptyString("userId", options.userId);

    this.adapter = options.adapter;
    this.embeddingProvider = options.embeddingProvider;
    this.config = resolveConfig({
      userId: options.userId,
      ...options.config,
    });
    this.engine = new CognitiveEngine(this.adapter, this.config);
  }

  /**
   * Store a new memory
   *
   * Generates embedding and initializes cognitive metadata.
   */
  async store(input: MemoryInput): Promise<string> {
    assertNonEmptyString("content", input.content);
    if (input.importance !== undefined)
      assertUnitInterval("importance", input.importance);
    if (input.stability !== undefined)
      assertUnitInterval("stability", input.stability);

    const embedding = await this.embedWithRetry(input.content);
    const category = input.category ?? input.memoryType ?? "semantic";

    const now = Date.now();
    const memory: Omit<Memory, "id" | "createdAt" | "updatedAt"> = {
      userId: this.config.userId,
      content: input.content,
      embedding,
      category,
      memoryType: category === "core" ? "semantic" : (category as any),
      importance: input.importance ?? this.config.defaultImportance,
      stability: input.stability ?? this.config.defaultStability,
      accessCount: 0,
      lastAccessed: now,
      retention: 1.0,
      metadata: input.metadata,
      associations: {},
      sessionIds: [],
      isCold: false,
      coldSince: null,
      daysAtFloor: 0,
      isSuperseded: false,
      supersededBy: null,
      isStub: false,
      contradictedBy: null,
      semanticType: input.semanticType ?? "other",
      validFrom: input.validFrom ?? null,
      validUntil: input.validUntil ?? null,
      ttlSeconds: input.ttlSeconds ?? null,
      sourceTurnIds: [],
    };

    return this.adapter.createMemory(memory);
  }

  /**
   * Extract memories from conversation text using LLM, check conflicts, and store.
   *
   * This is the main ingestion method matching Python's extract_and_store.
   */
  async extractAndStore(
    conversationText: string,
    sessionId: string,
    llm: LLMProvider,
  ): Promise<string[]> {
    assertNonEmptyString("conversationText", conversationText);
    assertNonEmptyString("sessionId", sessionId);

    const mode = this.config.extractionMode;
    if (!["raw", "semantic", "hybrid"].includes(mode)) {
      throw new Error(`Invalid extractionMode: "${mode}". Must be "raw", "semantic", or "hybrid".`);
    }

    const storedIds: string[] = [];

    // --- Semantic extraction (modes: semantic, hybrid) ---
    if (mode === "semantic" || mode === "hybrid") {
      const extracted = await extractFromConversation(
        conversationText,
        llm,
        this.config,
        sessionId,
      );

      for (const mem of extracted) {
        const embedding = await this.embedWithRetry(mem.content);

        const existing = await this.adapter.vectorSearch(embedding, {
          userId: this.config.userId,
          limit: 5,
        });

        let shouldStore = true;
        for (const existingMem of existing) {
          if (existingMem.relevanceScore > 0.85) {
            const conflict = await detectConflict(existingMem, mem, llm, this.config);
            if (conflict === "CONTRADICTION" || conflict === "UPDATE") {
              await this.adapter.updateMemory(existingMem.id, {
                content: mem.content,
                importance: Math.max(existingMem.importance, mem.importance),
                lastAccessed: Date.now(),
                contradictedBy: conflict === "CONTRADICTION" ? "new" : null,
              });
              storedIds.push(existingMem.id);
              shouldStore = false;
              break;
            } else if (conflict === "OVERLAP") {
              shouldStore = false;
              break;
            }
          }
        }

        if (shouldStore) {
          const id = await this.adapter.createMemory({
            ...mem,
            embedding,
          });
          storedIds.push(id);
        }
      }
    }

    // --- Raw turn storage (modes: raw, hybrid) ---
    if (mode === "raw" || mode === "hybrid") {
      const rawTurns = extractRawTurns(conversationText, this.config, sessionId);

      for (const mem of rawTurns) {
        const embedding = await this.embedWithRetry(mem.content);
        const id = await this.adapter.createMemory({
          ...mem,
          embedding,
        });
        storedIds.push(id);
      }
    }

    // Synaptic tagging — create associations between memories from same session
    if (storedIds.length > 1) {
      const storedMemories = await this.adapter.getMemories(storedIds);
      for (let i = 0; i < storedMemories.length; i++) {
        for (let j = i + 1; j < storedMemories.length; j++) {
          if (
            storedMemories[i].embedding.length > 0 &&
            storedMemories[j].embedding.length > 0
          ) {
            const sim = cosineSimilarity(
              storedMemories[i].embedding,
              storedMemories[j].embedding,
            );
            if (sim > 0.4) {
              await this.adapter.createOrStrengthenLink(
                storedMemories[i].id,
                storedMemories[j].id,
                sim * 0.5,
              );
            }
          }
        }
      }
    }

    // Optionally run maintenance
    if (this.config.runMaintenanceDuringIngestion) {
      await this.tick();
    }

    return storedIds;
  }

  /**
   * Search using the full 6-step retrieval pipeline from the engine.
   *
   * Returns SearchResponse with results, evidence chains, and optional trace.
   */
  async search(query: RetrievalQuery, llm?: LLMProvider): Promise<SearchResponse> {
    const {
      query: queryText,
      limit = 10,
      sessionId,
      deepRecall = false,
      trace = false,
    } = query;

    assertNonEmptyString("query", queryText);

    const queryEmbedding = await this.embedWithRetry(queryText);
    const now = Date.now();

    return this.engine.search(queryEmbedding, now, limit, sessionId, deepRecall, queryText, trace, llm);
  }

  /**
   * Retrieve memories relevant to a query (backward-compatible simpler API)
   *
   * Combines semantic similarity with retention weighting.
   */
  async retrieve(query: RetrievalQuery): Promise<ScoredMemory[]> {
    const {
      query: queryText,
      limit = 5,
      minRetention = this.config.minRetention,
      categories,
      memoryTypes,
      includeAssociations = true,
    } = query;

    assertNonEmptyString("query", queryText);

    const queryEmbedding = await this.embedWithRetry(queryText);

    const candidates = await this.adapter.vectorSearch(queryEmbedding, {
      userId: this.config.userId,
      categories,
      memoryTypes,
      minRetention,
      limit: limit * 3,
    });

    const scoredCandidates = candidates
      .map((m) => {
        const retention = this.engine.computeRetention(m);
        const finalScore = m.relevanceScore * retention;
        return { ...m, retention, finalScore };
      })
      .filter((m) => m.retention >= minRetention)
      .sort((a, b) => b.finalScore - a.finalScore)
      .slice(0, limit);

    const resultById = new Map<string, ScoredMemory>();
    for (const m of scoredCandidates) resultById.set(m.id, m);

    if (includeAssociations && scoredCandidates.length > 0) {
      const associated = await this.adapter.getLinkedMemoriesMultiple(
        scoredCandidates.map((m) => m.id),
        0.3,
      );

      for (const assoc of associated) {
        if (!resultById.has(assoc.id)) {
          const cosine = cosineSimilarity(queryEmbedding, assoc.embedding);
          const relevanceScore = Math.max(cosine, assoc.linkStrength);

          const retention = this.engine.computeRetention(assoc);
          if (retention < minRetention) continue;

          resultById.set(assoc.id, {
            ...assoc,
            retention,
            relevanceScore,
            finalScore: relevanceScore * retention,
          });
        }
      }
    }

    const results = Array.from(resultById.values())
      .sort((a, b) => b.finalScore - a.finalScore)
      .slice(0, limit);

    await this.strengthenMemories(results);
    await this.strengthenLinks(results.map((m) => m.id));

    return results;
  }

  /**
   * Get a memory by ID
   */
  async get(id: string): Promise<Memory | null> {
    const memory = await this.adapter.getMemory(id);

    if (memory) {
      await this.strengthenMemories([memory]);
    }

    return memory;
  }

  /**
   * Query memories for this user
   */
  async queryMemories(filters: MemoryFilters): Promise<Memory[]> {
    const memories = await this.adapter.queryMemories({
      ...filters,
      userId: this.config.userId,
    });
    if (memories.length > 0) {
      await this.strengthenMemories(memories);
    }
    return memories;
  }

  /**
   * Update a memory's content
   */
  async update(id: string, content: string): Promise<void> {
    assertNonEmptyString("content", content);

    const memory = await this.adapter.getMemory(id);
    if (!memory) {
      throw new Error(`Memory ${id} not found`);
    }

    const embedding = await this.embedWithRetry(content);

    await this.adapter.updateMemory(id, {
      content,
      embedding,
      updatedAt: Date.now(),
    });
  }

  /**
   * Run consolidation process
   */
  async consolidate(): Promise<ConsolidationResult> {
    const result: ConsolidationResult = {
      decayed: [],
      compressed: [],
      promotionCandidates: [],
      deleted: 0,
    };

    await this.refreshRetentionScores();

    // 1. Find fading memories
    const fading = await this.adapter.findFadingMemories(
      this.config.userId,
      this.config.consolidationRetentionThreshold,
    );

    result.decayed = fading.map((m) => ({
      id: m.id,
      retention: m.retention,
    }));

    // 2. Group by category, cluster by similarity
    const byCategory = new Map<string, Memory[]>();
    for (const memory of fading) {
      const cat = memory.category ?? "semantic";
      if (!byCategory.has(cat)) byCategory.set(cat, []);
      byCategory.get(cat)!.push(memory);
    }

    for (const [category, memories] of byCategory) {
      if (memories.length < this.config.consolidationGroupSize) continue;

      // Greedy clustering by embedding similarity
      const used = new Set<string>();
      const groups: Memory[][] = [];

      for (let i = 0; i < memories.length; i++) {
        if (used.has(memories[i].id)) continue;
        const group = [memories[i]];

        for (let j = i + 1; j < memories.length; j++) {
          if (used.has(memories[j].id)) continue;
          if (memories[i].embedding.length > 0 && memories[j].embedding.length > 0) {
            const sim = cosineSimilarity(memories[i].embedding, memories[j].embedding);
            if (sim >= this.config.consolidationSimilarityThreshold) {
              group.push(memories[j]);
              if (group.length >= this.config.consolidationGroupSize) break;
            }
          }
        }

        if (group.length >= this.config.consolidationGroupSize) {
          const finalGroup = group.slice(0, this.config.consolidationGroupSize);
          groups.push(finalGroup);
          for (const m of finalGroup) used.add(m.id);
        }
      }

      for (const group of groups) {
        const summary = this.summarizeMemories(group);

        const summaryId = await this.store({
          content: summary,
          category: category as any,
          importance: Math.max(...group.map((m) => m.importance)),
          metadata: {
            compressed: true,
            sourceCount: group.length,
            category,
          },
        });

        await this.adapter.markSuperseded(
          group.map((m) => m.id),
          summaryId,
        );

        result.compressed.push({
          summaryId,
          originalIds: group.map((m) => m.id),
          count: group.length,
        });
      }
    }

    // 3. Find promotion candidates
    const stable = await this.adapter.findStableMemories(
      this.config.userId,
      this.config.coreStabilityThreshold,
      this.config.coreAccessThreshold,
    );

    result.promotionCandidates = stable.map((m) => ({
      id: m.id,
      stability: m.stability,
      accessCount: m.accessCount,
    }));

    // 4. Delete very faded memories
    const veryFaded = await this.adapter.queryMemories({
      userId: this.config.userId,
      minRetention: 0,
    });

    const toDelete = veryFaded.filter((m) => {
      const daysSinceAccess =
        (Date.now() - m.lastAccessed) / (1000 * 60 * 60 * 24);
      return m.retention < 0.05 && daysSinceAccess > 30 && !m.isSuperseded;
    });

    if (toDelete.length > 0) {
      await this.adapter.deleteMemories(toDelete.map((m) => m.id));
      result.deleted = toDelete.length;
    }

    return result;
  }

  /**
   * Run all periodic maintenance: cold migration, TTL expiry, consolidation
   */
  async tick(llm?: LLMProvider): Promise<void> {
    const now = Date.now();
    const compressor = llm
      ? (contents: string[]) => compressMemories(contents, llm, this.config)
      : undefined;
    await this.engine.tick(now, this.embeddingProvider, compressor);
  }

  /**
   * Get memory system stats
   */
  async getStats(): Promise<MemoryStats> {
    const [hotCount, coldCount, stubCount, totalCount] = await Promise.all([
      this.adapter.hotCount(),
      this.adapter.coldCount(),
      this.adapter.stubCount(),
      this.adapter.totalCount(),
    ]);

    const allActive = await this.adapter.allActive();
    let coreCount = 0;
    let faintCount = 0;
    let totalRetention = 0;

    for (const m of allActive) {
      if (m.category === "core") coreCount++;
      const retention = this.engine.computeRetention(m);
      if (retention < this.config.faintThreshold) faintCount++;
      totalRetention += retention;
    }

    return {
      total: totalCount,
      hot: hotCount,
      cold: coldCount,
      stub: stubCount,
      core: coreCount,
      faint: faintCount,
      avgRetention: allActive.length > 0 ? totalRetention / allActive.length : 0,
    };
  }

  /**
   * Clear all memories for this user
   */
  async clear(): Promise<void> {
    await this.adapter.clear();
  }

  /**
   * Recompute + persist retention for all memories for this user.
   */
  async refreshRetentionScores(): Promise<void> {
    const memories = await this.adapter.queryMemories({
      userId: this.config.userId,
      minRetention: 0,
      includeSuperseded: true,
    });

    const updates = new Map<string, number>();
    for (const m of memories) {
      const retention = this.engine.computeRetention(m);
      updates.set(m.id, retention);
    }
    if (updates.size > 0) await this.adapter.updateRetentionScores(updates);
  }

  /**
   * Create a link between two memories
   */
  async link(
    sourceId: string,
    targetId: string,
    strength: number = 0.5,
  ): Promise<void> {
    assertUnitInterval("strength", strength);
    await this.adapter.createOrStrengthenLink(sourceId, targetId, strength);
  }

  // ------------------------------------------------------------------
  // Private methods
  // ------------------------------------------------------------------

  private async strengthenMemories(memories: Memory[]): Promise<void> {
    const now = Date.now();

    const updates: Array<{ id: string; updates: Partial<Memory> }> = [];

    for (const memory of memories) {
      assertUnitInterval("stability", memory.stability);
      assertUnitInterval("importance", memory.importance);
      assertNonNegativeInt("accessCount", memory.accessCount);
      if (!Number.isFinite(memory.lastAccessed)) {
        throw new Error(
          `Invalid lastAccessed: ${memory.lastAccessed} (must be valid timestamp)`,
        );
      }

      const daysSinceAccess =
        (now - memory.lastAccessed) / (1000 * 60 * 60 * 24);

      const newStability = updateStability(
        memory.stability,
        daysSinceAccess,
        this.config.directBoost,
        this.config.maxSpacedRepMultiplier,
        this.config.spacedRepIntervalDays,
      );

      const newRetention = this.engine.computeRetention({
        ...memory,
        stability: newStability,
        accessCount: memory.accessCount + 1,
        lastAccessed: now,
      });

      updates.push({
        id: memory.id,
        updates: {
          stability: newStability,
          accessCount: memory.accessCount + 1,
          lastAccessed: now,
          retention: newRetention,
        },
      });
    }

    await this.adapter.transaction(async (adapter) => {
      await Promise.all(
        updates.map(({ id, updates: memoryUpdates }) =>
          adapter.updateMemory(id, memoryUpdates),
        ),
      );
    });
  }

  private async strengthenLinks(memoryIds: string[]): Promise<void> {
    for (let i = 0; i < memoryIds.length; i++) {
      for (let j = i + 1; j < memoryIds.length; j++) {
        await this.adapter.createOrStrengthenLink(
          memoryIds[i],
          memoryIds[j],
          0.1,
        );
      }
    }
  }

  private summarizeMemories(memories: Memory[]): string {
    const combined = memories.map((m) => m.content).join(". ");
    return combined.length > 500 ? `${combined.slice(0, 497)}...` : combined;
  }

  private async embedWithRetry(text: string): Promise<number[]> {
    let lastError: unknown;
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        return await this.embeddingProvider.embed(text);
      } catch (err) {
        lastError = err;
        if (attempt < 2) {
          const delayMs = Math.min(2000, 250 * 2 ** attempt);
          await sleep(delayMs);
        }
      }
    }
    throw new Error(`Embedding failed: ${String(lastError)}`);
  }
}

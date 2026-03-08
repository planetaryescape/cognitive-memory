/**
 * Cognitive Memory System - LLM Extraction & Conflict Detection
 *
 * Uses an LLM to:
 * 1. Extract discrete memories from conversation turns
 * 2. Classify memory category (episodic/semantic/procedural/core)
 * 3. Assign importance scores
 * 4. Detect conflicts with existing memories
 * 5. Compress memory groups during consolidation
 */

import type { Memory, MemoryCategory, ResolvedCognitiveMemoryConfig } from "./types";
import { createDefaultMemory } from "./types";

// ---------------------------------------------------------------------------
// LLM Provider interface
// ---------------------------------------------------------------------------

export interface LLMProvider {
  complete(
    prompt: string,
    options?: {
      model?: string;
      maxTokens?: number;
      temperature?: number;
    },
  ): Promise<string>;
}

// ---------------------------------------------------------------------------
// Prompts (ported from Python SDK)
// ---------------------------------------------------------------------------

export const EXTRACTION_PROMPT = `Extract ALL facts and events from this conversation. Be thorough — extract every distinct piece of information, no matter how brief or incidental.

You are a NARRATOR, not a summarizer. Record what happened and what was said, not your interpretation of it.

For each memory, provide:
- content: one specific fact or event in a clear sentence. INCLUDE specific names, dates, numbers, and places.
- category:
  - "core": identity info (name, age, gender, relationship status, nationality, medical, family members, profession, where they live/moved from)
  - "semantic": lasting facts, preferences, plans, relationships, opinions, hobbies. DEFAULT if unsure.
  - "episodic": specific one-time events with dates/times
  - "procedural": routines, habits, skills
- importance: 0.0 to 1.0

CRITICAL RULES:
1. NARRATE, don't interpret. Store WHAT HAPPENED, not what it means.
   BAD: "Alex enjoys outdoor activities" (interpretation)
   GOOD: "Alex went hiking at Mount Rainier on March 12, 2024" (what happened)
   BAD: "Sam is artistic" (interpretation)
   GOOD: "Sam painted a landscape of the lake in 2023" (what happened)
2. Extract EVERY specific event, activity, and experience mentioned — even brief ones. A picnic, a book read, a race run, a song listened to — ALL get their own memory.
3. RESOLVE relative dates using the conversation date at the top (e.g., conversation on "8 May 2023" + "yesterday" = May 7, 2023). Include resolved dates in the content.
4. For lasting facts (preferences, traits, relationships), extract those too as semantic memories.
5. Extract each distinct fact as a SEPARATE memory. One fact per memory.
6. If messages are labeled User and Assistant, PRIORITIZE extracting memories from User messages. User messages contain personal information we need to remember. Assistant messages are less important unless they contain facts the user confirmed.
7. Don't skip brief or passing mentions. If someone mentions a fact once in a single sentence, it's still a memory worth storing. A passing reference to a hometown, a book title, or a pet's name is just as important as a detailed story.

Conversation:
{conversation}

Respond with a JSON array only. No markdown, no preamble.
Example: [{"content": "Alex is a 32-year-old software engineer", "category": "core", "importance": 0.9}, {"content": "Alex is single", "category": "core", "importance": 0.7}, {"content": "Alex moved from Germany three years ago", "category": "core", "importance": 0.8}, {"content": "Sam has two dogs named Biscuit and Maple", "category": "core", "importance": 0.8}, {"content": "Alex finished reading The Great Gatsby in January 2024", "category": "episodic", "importance": 0.5}, {"content": "Sam ran a 5K for charity the weekend before March 10, 2024", "category": "episodic", "importance": 0.5}]`;

export const CONFLICT_PROMPT = `Does the new memory contradict or update an existing memory?

Existing memory: "{existing}"
New memory: "{new}"

Respond with exactly one word: CONTRADICTION, UPDATE, OVERLAP, or NONE.
- CONTRADICTION: the new memory directly negates the existing one
- UPDATE: the new memory is a newer version of the same fact
- OVERLAP: they cover similar ground but don't conflict
- NONE: they are unrelated`;

export const CONSOLIDATION_PROMPT = `Compress these related memories into a single concise summary that preserves all key facts.

Memories:
{memories}

Write one clear paragraph. Preserve specific names, dates, numbers, and preferences. Do not add information that isn't in the originals.`;

// ---------------------------------------------------------------------------
// Conflict types
// ---------------------------------------------------------------------------

export type ConflictType = "CONTRADICTION" | "UPDATE" | "OVERLAP" | "NONE";

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

const VALID_CATEGORIES = new Set<MemoryCategory>(["episodic", "semantic", "procedural", "core"]);

export async function extractFromConversation(
  conversationText: string,
  llm: LLMProvider,
  config: ResolvedCognitiveMemoryConfig,
  sessionId: string,
): Promise<Array<Omit<Memory, "id" | "createdAt" | "updatedAt" | "embedding">>> {
  let prompt = EXTRACTION_PROMPT.replace("{conversation}", conversationText);

  if (config.customExtractionInstructions) {
    prompt = `IMPORTANT INSTRUCTIONS FOR MEMORY EXTRACTION:\n${config.customExtractionInstructions}\n\n${prompt}`;
  }

  const raw = await llm.complete(prompt, {
    model: config.extractionModel,
    maxTokens: 2000,
    temperature: 0,
  });

  const items = parseExtractionResponse(raw, conversationText);
  return buildMemories(items, config, sessionId);
}

function parseExtractionResponse(
  raw: string,
  fallbackText: string,
): Array<Record<string, unknown>> {
  try {
    let cleaned = raw.trim();
    if (cleaned.startsWith("```")) {
      cleaned = cleaned.replace(/^```(?:json)?\s*/, "").replace(/\s*```$/, "");
    }
    const parsed = JSON.parse(cleaned);
    if (Array.isArray(parsed)) return parsed;
    return [{ content: fallbackText.slice(0, 500), category: "episodic", importance: 0.5 }];
  } catch {
    return [{ content: fallbackText.slice(0, 500), category: "episodic", importance: 0.5 }];
  }
}

function buildMemories(
  items: Array<Record<string, unknown>>,
  config: ResolvedCognitiveMemoryConfig,
  sessionId: string,
): Array<Omit<Memory, "id" | "createdAt" | "updatedAt" | "embedding">> {
  const now = Date.now();
  const memories: Array<Omit<Memory, "id" | "createdAt" | "updatedAt" | "embedding">> = [];

  for (const item of items) {
    if (typeof item !== "object" || item === null) continue;

    const content = typeof item.content === "string" ? item.content.trim() : "";
    if (!content) continue;

    const catStr = (typeof item.category === "string" ? item.category.toLowerCase() : "semantic") as MemoryCategory;
    const category: MemoryCategory = VALID_CATEGORIES.has(catStr) ? catStr : "semantic";

    let importance = typeof item.importance === "number" ? item.importance : 0.5;
    importance = Math.max(0.0, Math.min(1.0, importance));

    const stability = 0.1 + importance * 0.3;

    memories.push({
      userId: config.userId,
      content,
      category,
      memoryType: category === "core" ? "semantic" : (category as any),
      importance,
      stability,
      accessCount: 0,
      lastAccessed: now,
      retention: 1.0,
      associations: {},
      sessionIds: [sessionId],
      isCold: false,
      coldSince: null,
      daysAtFloor: 0,
      isSuperseded: false,
      supersededBy: null,
      isStub: false,
      contradictedBy: null,
    });
  }

  return memories;
}

// ---------------------------------------------------------------------------
// Raw turn extraction (no LLM)
// ---------------------------------------------------------------------------

export function extractRawTurns(
  conversationText: string,
  config: ResolvedCognitiveMemoryConfig,
  sessionId: string,
): Array<Omit<Memory, "id" | "createdAt" | "updatedAt" | "embedding">> {
  const now = Date.now();
  const lines = conversationText.trim().split("\n");
  const memories: Array<Omit<Memory, "id" | "createdAt" | "updatedAt" | "embedding">> = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;
    // Skip header lines like "[This conversation took place on ...]"
    if (line.startsWith("[") && line.endsWith("]")) continue;

    memories.push({
      userId: config.userId,
      content: line,
      category: "episodic" as MemoryCategory,
      memoryType: "episodic" as any,
      importance: 0.5,
      stability: 0.2,
      accessCount: 0,
      lastAccessed: now,
      retention: 1.0,
      associations: {},
      sessionIds: [sessionId],
      isCold: false,
      coldSince: null,
      daysAtFloor: 0,
      isSuperseded: false,
      supersededBy: null,
      isStub: false,
      contradictedBy: null,
    });
  }

  return memories;
}

// ---------------------------------------------------------------------------
// Conflict detection
// ---------------------------------------------------------------------------

export async function detectConflict(
  existingMemory: Memory,
  newMemory: { content: string },
  llm: LLMProvider,
  config: ResolvedCognitiveMemoryConfig,
): Promise<ConflictType> {
  const prompt = CONFLICT_PROMPT
    .replace("{existing}", existingMemory.content)
    .replace("{new}", newMemory.content);

  const raw = await llm.complete(prompt, {
    model: config.extractionModel,
    maxTokens: 20,
    temperature: 0,
  });

  const upper = raw.trim().toUpperCase();
  for (const label of ["CONTRADICTION", "UPDATE", "OVERLAP", "NONE"] as ConflictType[]) {
    if (upper.includes(label)) return label;
  }
  return "NONE";
}

// ---------------------------------------------------------------------------
// Memory compression (for consolidation)
// ---------------------------------------------------------------------------

export async function compressMemories(
  contents: string[],
  llm: LLMProvider,
  config: ResolvedCognitiveMemoryConfig,
): Promise<string> {
  const numbered = contents.map((c, i) => `${i + 1}. ${c}`).join("\n");
  const prompt = CONSOLIDATION_PROMPT.replace("{memories}", numbered);

  return llm.complete(prompt, {
    model: config.extractionModel,
    maxTokens: 500,
    temperature: 0,
  });
}

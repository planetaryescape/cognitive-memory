// Adapters
export type { ConvexAdapterFunctions, MemoryFilters } from "./adapters";
export { ConvexAdapter, InMemoryAdapter, MemoryAdapter } from "./adapters";

// Core types
export type {
  Association,
  CognitiveMemoryConfig,
  ConsolidationResult,
  DecayParameters,
  EmbeddingProvider,
  Memory,
  MemoryCategory,
  MemoryInput,
  MemoryLink,
  MemoryStats,
  MemoryType,
  ResolvedCognitiveMemoryConfig,
  RetrievalQuery,
  ScoredMemory,
  SearchResult,
} from "./core";

// Core classes and functions
export {
  BASE_DECAY_RATES,
  DECAY_FLOORS,
  CognitiveEngine,
  CognitiveMemory,
  DEFAULT_CONFIG,
  HashEmbeddingProvider,
  OpenAIEmbeddingProvider,
  calculateRetention,
  createDefaultMemory,
  getBaseDecayRate,
  getRetentionFloor,
  resolveConfig,
  updateStability,
} from "./core";

// Extraction
export {
  CONFLICT_PROMPT,
  CONSOLIDATION_PROMPT,
  EXTRACTION_PROMPT,
  compressMemories,
  detectConflict,
  extractFromConversation,
} from "./core";
export type { ConflictType, LLMProvider } from "./core";

// Embedding providers
export type {
  HashEmbeddingProviderOptions,
  OpenAIEmbeddingProviderOptions,
} from "./core";

// Utils
export {
  categorizeMemoryType,
  cosineSimilarity,
  euclideanDistance,
  extractTopics,
  normalizeVector,
  scoreImportance,
} from "./utils";

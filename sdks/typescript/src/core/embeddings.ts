/**
 * Cognitive Memory System - Embedding Providers
 *
 * Concrete implementations of EmbeddingProvider:
 * - OpenAIEmbeddingProvider: Uses OpenAI embeddings API via raw fetch
 * - HashEmbeddingProvider: Deterministic hash-based embeddings for testing
 */

import { setTimeout as sleep } from "node:timers/promises";
import type { EmbeddingProvider } from "./types";

// ---------------------------------------------------------------------------
// OpenAI Embedding Provider
// ---------------------------------------------------------------------------

export interface OpenAIEmbeddingProviderOptions {
  apiKey: string;
  model?: string;
  dimensions?: number;
  baseUrl?: string;
}

export class OpenAIEmbeddingProvider implements EmbeddingProvider {
  private apiKey: string;
  private model: string;
  private dims: number;
  private baseUrl: string;

  readonly dimensions: number;

  constructor(options: OpenAIEmbeddingProviderOptions) {
    this.apiKey = options.apiKey;
    this.model = options.model ?? "text-embedding-3-small";
    this.dims = options.dimensions ?? 1536;
    this.dimensions = this.dims;
    this.baseUrl = options.baseUrl ?? "https://api.openai.com/v1";
  }

  async embed(text: string): Promise<number[]> {
    const result = await this.embedBatch([text]);
    return result[0];
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    let lastError: unknown;

    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const response = await fetch(`${this.baseUrl}/embeddings`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${this.apiKey}`,
          },
          body: JSON.stringify({
            model: this.model,
            input: texts,
            dimensions: this.dims,
          }),
        });

        if (!response.ok) {
          const body = await response.text();
          throw new Error(
            `OpenAI embeddings API error ${response.status}: ${body}`,
          );
        }

        const data = (await response.json()) as {
          data: Array<{ embedding: number[]; index: number }>;
        };

        // Sort by index to maintain order
        const sorted = data.data.sort((a, b) => a.index - b.index);
        return sorted.map((d) => d.embedding);
      } catch (err) {
        lastError = err;
        if (attempt < 2) {
          await sleep(Math.min(2000, 250 * 2 ** attempt));
        }
      }
    }

    throw new Error(`OpenAI embedding failed after 3 attempts: ${String(lastError)}`);
  }
}

// ---------------------------------------------------------------------------
// Hash Embedding Provider (for testing)
// ---------------------------------------------------------------------------

export interface HashEmbeddingProviderOptions {
  dimensions?: number;
}

/**
 * Deterministic hash-based embedding provider for testing.
 * Produces consistent pseudo-random unit vectors from text input.
 * No external dependencies.
 */
export class HashEmbeddingProvider implements EmbeddingProvider {
  readonly dimensions: number;

  constructor(options?: HashEmbeddingProviderOptions) {
    this.dimensions = options?.dimensions ?? 128;
  }

  async embed(text: string): Promise<number[]> {
    return this.hashToVector(text);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return texts.map((t) => this.hashToVector(t));
  }

  private hashToVector(text: string): number[] {
    // Simple hash seeding using FNV-1a-like approach
    let h1 = 0x811c9dc5;
    let h2 = 0x01000193;
    for (let i = 0; i < text.length; i++) {
      h1 = Math.imul(h1 ^ text.charCodeAt(i), 0x01000193);
      h2 = Math.imul(h2 ^ text.charCodeAt(i), 0x811c9dc5);
    }

    // Generate pseudo-random vector using xorshift
    const vec: number[] = new Array(this.dimensions);
    let seed = (h1 ^ h2) >>> 0;

    for (let i = 0; i < this.dimensions; i++) {
      seed ^= seed << 13;
      seed ^= seed >> 17;
      seed ^= seed << 5;
      // Map to [-1, 1]
      vec[i] = ((seed >>> 0) / 0xffffffff) * 2 - 1;
    }

    // Normalize to unit vector
    let magnitude = 0;
    for (let i = 0; i < this.dimensions; i++) {
      magnitude += vec[i] * vec[i];
    }
    magnitude = Math.sqrt(magnitude);

    if (magnitude > 0) {
      for (let i = 0; i < this.dimensions; i++) {
        vec[i] /= magnitude;
      }
    }

    return vec;
  }
}

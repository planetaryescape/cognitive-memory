import { cosineSimilarity } from "../utils/embeddings";
import type { Memory } from "./types";

/**
 * Greedy clustering by embedding similarity.
 * Groups memories where each member is within `simThreshold` cosine similarity
 * of the group seed. Only returns groups that reach `groupSize`.
 */
export function greedyClusterBySimilarity(
  memories: Memory[],
  simThreshold: number,
  groupSize: number,
): Memory[][] {
  const used = new Set<string>();
  const groups: Memory[][] = [];

  for (let i = 0; i < memories.length; i++) {
    if (used.has(memories[i].id)) continue;
    const group = [memories[i]];

    for (let j = i + 1; j < memories.length; j++) {
      if (used.has(memories[j].id)) continue;
      if (memories[i].embedding.length > 0 && memories[j].embedding.length > 0) {
        const sim = cosineSimilarity(memories[i].embedding, memories[j].embedding);
        if (sim >= simThreshold) {
          group.push(memories[j]);
          if (group.length >= groupSize) break;
        }
      }
    }

    if (group.length >= groupSize) {
      const finalGroup = group.slice(0, groupSize);
      groups.push(finalGroup);
      for (const m of finalGroup) used.add(m.id);
    }
  }

  return groups;
}

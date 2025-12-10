package io.github.vishalmysore;

import java.util.List;

/**
 * Interface for implementing custom chunking strategies in RAG systems.
 * 
 * Chunking strategies determine how to split large documents into smaller,
 * semantically meaningful units for indexing and retrieval.
 * 
 * Implementations should be stateless and thread-safe.
 * 
 * Example usage:
 * <pre>
 * ChunkingStrategy strategy = new SlidingWindowChunking(150, 30);
 * List&lt;String&gt; chunks = strategy.chunk(documentContent);
 * </pre>
 */
public interface ChunkingStrategy {
    
    /**
     * Splits the input content into chunks according to this strategy's algorithm.
     * 
     * @param content The text content to be chunked
     * @return A list of text chunks. Each chunk should be a standalone, meaningful unit.
     *         Empty list if content is null or empty.
     */
    List<String> chunk(String content);
    
    /**
     * Returns a human-readable name for this chunking strategy.
     * 
     * @return The strategy name (e.g., "Sliding Window", "Entity-Based")
     */
    String getName();
    
    /**
     * Returns a detailed description of how this strategy works.
     * 
     * @return A description explaining the chunking algorithm and its use cases
     */
    String getDescription();
}

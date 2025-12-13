package io.github.vishalmysore.rag;

/**
 * Extended interface for embedding providers that support token-level embeddings.
 * Required for Late Chunking strategy which embeds the entire document first,
 * then applies chunking boundaries to the token-level embeddings.
 * 
 * This approach preserves full document context in each chunk embedding,
 * unlike traditional chunk-first-then-embed approaches that lose cross-chunk context.
 */
public interface TokenLevelEmbeddingProvider extends EmbeddingProvider {
    
    /**
     * Generates token-level embeddings for the entire input text.
     * Each token's embedding is contextualized by the full document.
     * 
     * @param text The complete text to embed at token level
     * @return A 2D array where each row is a token's embedding vector,
     *         preserving full document context. Array dimensions are [numTokens][embeddingDim]
     */
    float[][] embedTokenLevel(String text);
    
    /**
     * Gets the tokenization of the input text, matching the token-level embeddings.
     * This allows mapping between text segments and their corresponding token embeddings.
     * 
     * @param text The text to tokenize
     * @return Array of tokens in the order they appear in embedTokenLevel() output
     */
    String[] tokenize(String text);
    
    /**
     * Gets the maximum context length (in tokens) that this provider can process.
     * For Late Chunking, larger context windows allow better preservation of document semantics.
     * 
     * @return Maximum number of tokens that can be processed in a single call
     */
    int getMaxContextLength();
}

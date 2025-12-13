package io.github.vishalmysore.rag;

/**
 * Interface for generating contextual descriptions or summaries.
 * Used to enrich chunks with document-level context before embedding.
 * 
 * Implementations typically use an LLM to generate a descriptive blurb
 * that situates the chunk within the broader document context.
 */
public interface ContextGenerator {
    
    /**
     * Generates a contextual description for a chunk within a document.
     * This context is typically prepended to the chunk before embedding,
     * enriching it with document-level awareness.
     * 
     * @param fullDocument The complete document text
     * @param chunk The specific chunk to generate context for
     * @param chunkIndex The position of this chunk in the document (0-based)
     * @param totalChunks Total number of chunks in the document
     * @return A contextual description or summary to prepend to the chunk
     */
    String generateContext(String fullDocument, String chunk, int chunkIndex, int totalChunks);
    
    /**
     * Generates a high-level document summary that can be used across multiple chunks.
     * This is more efficient than generating per-chunk context when the same
     * document-level context applies to all chunks.
     * 
     * @param fullDocument The complete document text
     * @return A document summary to prepend to each chunk
     */
    default String generateDocumentSummary(String fullDocument) {
        return "Document context: " + fullDocument.substring(0, Math.min(200, fullDocument.length())) + "...";
    }
}

package io.github.vishalmysore.rag.chunking;

import io.github.vishalmysore.rag.ChunkingStrategy;
import io.github.vishalmysore.rag.ContextGenerator;

import java.util.ArrayList;
import java.util.List;

/**
 * Contextual Chunking Strategy
 * 
 * Enriches chunks with document-level context before embedding. Each chunk is prepended
 * with an LLM-generated summary or descriptive blurb that situates the chunk within the
 * broader document context.
 * 
 * This is a decorator/wrapper pattern - it wraps any existing ChunkingStrategy and adds
 * contextual enrichment on top. The enriched chunks maintain awareness of the full document,
 * improving retrieval accuracy even when using traditional embed-after-chunk approaches.
 * 
 * How it works:
 * 1. Uses the wrapped strategy to chunk the document
 * 2. For each chunk, generates contextual description using LLM or heuristics
 * 3. Prepends the context to each chunk
 * 4. Returns enriched chunks that can be embedded independently
 * 
 * Unlike Late Chunking (which embeds first, chunks later), Contextual Chunking follows
 * the traditional chunk-first approach but enhances each chunk with generated context.
 * 
 * Best for: Any scenario where you want chunks to retain document awareness but cannot
 * use token-level embeddings (Late Chunking). Works with any embedding model.
 * 
 * Example:
 * <pre>
 * ChunkingStrategy baseStrategy = new SlidingWindowChunking(200, 40);
 * ContextGenerator contextGen = new SimpleContextGenerator();
 * ChunkingStrategy enrichedStrategy = new ContextualChunking(baseStrategy, contextGen);
 * 
 * List&lt;String&gt; enrichedChunks = enrichedStrategy.chunk(document);
 * // Each chunk now has: "[CONTEXT]\n\n[ORIGINAL CHUNK]"
 * </pre>
 */
public class ContextualChunking implements ChunkingStrategy {
    
    private final ChunkingStrategy baseStrategy;
    private final ContextGenerator contextGenerator;
    private final boolean prependContext;
    
    // Cache for last chunking operation
    private String cachedContent;
    private List<String> cachedBaseChunks;
    private List<String> cachedEnrichedChunks;
    
    /**
     * Creates a Contextual Chunking strategy that wraps an existing strategy.
     * Context is prepended to each chunk by default.
     * 
     * @param baseStrategy The underlying chunking strategy to wrap
     * @param contextGenerator Generator for contextual descriptions
     */
    public ContextualChunking(ChunkingStrategy baseStrategy, ContextGenerator contextGenerator) {
        this(baseStrategy, contextGenerator, true);
    }
    
    /**
     * Creates a Contextual Chunking strategy with custom context placement.
     * 
     * @param baseStrategy The underlying chunking strategy to wrap
     * @param contextGenerator Generator for contextual descriptions
     * @param prependContext If true, prepend context; if false, append context
     */
    public ContextualChunking(ChunkingStrategy baseStrategy, 
                             ContextGenerator contextGenerator,
                             boolean prependContext) {
        if (baseStrategy == null) {
            throw new IllegalArgumentException("Base strategy cannot be null");
        }
        if (contextGenerator == null) {
            throw new IllegalArgumentException("Context generator cannot be null");
        }
        
        this.baseStrategy = baseStrategy;
        this.contextGenerator = contextGenerator;
        this.prependContext = prependContext;
    }
    
    @Override
    public List<String> chunk(String content) {
        if (content == null || content.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        // Step 1: Use the base strategy to chunk the document
        cachedContent = content;
        cachedBaseChunks = baseStrategy.chunk(content);
        
        // Step 2: Enrich each chunk with contextual information
        cachedEnrichedChunks = new ArrayList<>();
        int totalChunks = cachedBaseChunks.size();
        
        for (int i = 0; i < totalChunks; i++) {
            String baseChunk = cachedBaseChunks.get(i);
            
            // Generate contextual description for this chunk
            String context = contextGenerator.generateContext(
                content, 
                baseChunk, 
                i, 
                totalChunks
            );
            
            // Combine context and chunk
            String enrichedChunk;
            if (prependContext) {
                enrichedChunk = context + "\n\n" + baseChunk;
            } else {
                enrichedChunk = baseChunk + "\n\n" + context;
            }
            
            cachedEnrichedChunks.add(enrichedChunk);
        }
        
        return cachedEnrichedChunks;
    }
    
    /**
     * Gets the original chunks without contextual enrichment.
     * Useful for comparison or when you want to see the base chunks.
     * 
     * @return List of chunks from the base strategy, without context
     */
    public List<String> getBaseChunks() {
        if (cachedBaseChunks == null) {
            throw new IllegalStateException("Must call chunk() before getBaseChunks()");
        }
        return new ArrayList<>(cachedBaseChunks);
    }
    
    /**
     * Gets the chunks with contextual enrichment (same as chunk() return value).
     * Provided for clarity and consistency with other chunking strategies.
     * 
     * @return List of enriched chunks with contextual headers
     */
    public List<String> getEnrichedChunks() {
        if (cachedEnrichedChunks == null) {
            throw new IllegalStateException("Must call chunk() before getEnrichedChunks()");
        }
        return new ArrayList<>(cachedEnrichedChunks);
    }
    
    /**
     * Gets the underlying base chunking strategy.
     * 
     * @return The wrapped chunking strategy
     */
    public ChunkingStrategy getBaseStrategy() {
        return baseStrategy;
    }
    
    /**
     * Gets the context generator being used.
     * 
     * @return The context generator
     */
    public ContextGenerator getContextGenerator() {
        return contextGenerator;
    }
    
    @Override
    public String getName() {
        return "Contextual Chunking (" + baseStrategy.getName() + ")";
    }
    
    @Override
    public String getDescription() {
        return "Enriches chunks with LLM-generated document context before embedding. " +
               "Wraps: " + baseStrategy.getDescription() + " " +
               "Each chunk is " + (prependContext ? "prepended" : "appended") + 
               " with a contextual description that situates it within the full document, " +
               "improving retrieval accuracy for queries requiring document-level understanding.";
    }
}

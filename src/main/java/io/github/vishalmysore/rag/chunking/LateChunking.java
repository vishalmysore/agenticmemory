package io.github.vishalmysore.rag.chunking;

import io.github.vishalmysore.rag.ChunkingStrategy;
import io.github.vishalmysore.rag.TokenLevelEmbeddingProvider;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Late Chunking Strategy
 * Based on the "Reconstructing Context Evaluating Advanced Chunking Strategies for Retrieval-Augmented Generation " paper. https://arxiv.org/pdf/2504.19754 
 * An advanced method for preparing long documents for retrieval systems, designed to overcome
 * the critical problem of context loss that occurs in traditional document processing.
 * 
 * Unlike the standard approach, which first splits a document into small, isolated chunks and
 * then embeds each chunk independently, Late Chunking reverses the order. It first leverages
 * a specialized, long-context embedding model to process the entire document (or the largest
 * possible segment) in a single pass. This single-pass embedding generates a sequence of
 * token-level vectors, where each token's vector representation is infused with the global,
 * full-document context and semantic relationships from every other part of the text.
 * 
 * The "chunking" then happens after the embedding is complete, hence the term "Late." The system
 * applies pre-defined segment boundaries (based on logical divisions like sentences, paragraphs,
 * or fixed token spans) to the long sequence of context-rich token embeddings. For each defined
 * segment, a simple pooling technique (such as mean pooling, which averages the vectors) is used
 * to combine the token vectors within that segment into a single, final chunk embedding vector.
 * 
 * The result is a set of smaller, retrievable chunk embeddings that have all the benefits of
 * their compact size but still retain the deep, accurate semantic context of the entire original
 * document. This significantly improves retrieval accuracy, especially for queries that rely on
 * long-range dependencies or cross-chunk references.
 * 
 * Best for: Long documents with cross-references, technical documentation, research papers,
 * legal documents where context preservation is critical
 * 
 * Parameters:
 * - boundaryType: SENTENCE, PARAGRAPH, or FIXED_TOKENS
 * - tokensPerChunk: For FIXED_TOKENS mode, number of tokens per chunk
 * - poolingStrategy: MEAN (average), MAX, or FIRST (use first token)
 */
public class LateChunking implements ChunkingStrategy {
    
    /**
     * Defines how to determine chunk boundaries in the token sequence
     */
    public enum BoundaryType {
        /** Split on sentence boundaries (periods, question marks, exclamation points) */
        SENTENCE,
        /** Split on paragraph boundaries (double newlines) */
        PARAGRAPH,
        /** Split on fixed token count intervals */
        FIXED_TOKENS
    }
    
    /**
     * Defines how to pool token embeddings within a chunk
     */
    public enum PoolingStrategy {
        /** Average all token embeddings (most common, balances all tokens) */
        MEAN,
        /** Take the maximum value across all tokens for each dimension */
        MAX,
        /** Use only the first token's embedding (like BERT's [CLS] token) */
        FIRST
    }
    
    private final BoundaryType boundaryType;
    private final int tokensPerChunk;
    private final PoolingStrategy poolingStrategy;
    private final TokenLevelEmbeddingProvider embeddingProvider;
    
    // Cached token embeddings for the current document
    private float[][] cachedTokenEmbeddings;
    private String[] cachedTokens;
    private String cachedContent;
    
    /**
     * Creates a Late Chunking strategy with sentence boundaries and mean pooling.
     * 
     * @param embeddingProvider Provider that supports token-level embeddings
     */
    public LateChunking(TokenLevelEmbeddingProvider embeddingProvider) {
        this(embeddingProvider, BoundaryType.SENTENCE, 128, PoolingStrategy.MEAN);
    }
    
    /**
     * Creates a Late Chunking strategy with custom settings.
     * 
     * @param embeddingProvider Provider that supports token-level embeddings
     * @param boundaryType How to determine chunk boundaries
     * @param tokensPerChunk Number of tokens per chunk (only used for FIXED_TOKENS mode)
     * @param poolingStrategy How to combine token embeddings within each chunk
     */
    public LateChunking(TokenLevelEmbeddingProvider embeddingProvider, 
                       BoundaryType boundaryType,
                       int tokensPerChunk,
                       PoolingStrategy poolingStrategy) {
        this.embeddingProvider = embeddingProvider;
        this.boundaryType = boundaryType;
        this.tokensPerChunk = tokensPerChunk;
        this.poolingStrategy = poolingStrategy;
    }
    
    @Override
    public List<String> chunk(String content) {
        if (content == null || content.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        // Step 1: Embed the entire document at token level (preserving full context)
        // This is the key difference - we embed BEFORE chunking
        cachedContent = content;
        cachedTokenEmbeddings = embeddingProvider.embedTokenLevel(content);
        cachedTokens = embeddingProvider.tokenize(content);
        
        // Step 2: Determine chunk boundaries based on the configured strategy
        List<ChunkBoundary> boundaries = determineBoundaries(content, cachedTokens);
        
        // Step 3: Create text chunks based on boundaries
        List<String> chunks = new ArrayList<>();
        for (ChunkBoundary boundary : boundaries) {
            String chunkText = extractChunkText(content, cachedTokens, boundary);
            if (!chunkText.trim().isEmpty()) {
                chunks.add(chunkText);
            }
        }
        
        return chunks;
    }
    
    /**
     * Gets the pooled embedding for a specific chunk.
     * This method should be called after chunk() to get context-rich embeddings.
     * 
     * @param chunkIndex The index of the chunk (0-based)
     * @return The pooled embedding vector for this chunk, with full document context
     */
    public float[] getChunkEmbedding(int chunkIndex) {
        if (cachedTokenEmbeddings == null) {
            throw new IllegalStateException("Must call chunk() before getChunkEmbedding()");
        }
        
        List<ChunkBoundary> boundaries = determineBoundaries(cachedContent, cachedTokens);
        if (chunkIndex < 0 || chunkIndex >= boundaries.size()) {
            throw new IndexOutOfBoundsException("Chunk index out of range: " + chunkIndex);
        }
        
        ChunkBoundary boundary = boundaries.get(chunkIndex);
        return poolTokenEmbeddings(cachedTokenEmbeddings, boundary.startToken, boundary.endToken);
    }
    
    /**
     * Gets all chunk embeddings at once.
     * 
     * @return Array of chunk embedding vectors, each preserving full document context
     */
    public float[][] getAllChunkEmbeddings() {
        if (cachedTokenEmbeddings == null) {
            throw new IllegalStateException("Must call chunk() before getAllChunkEmbeddings()");
        }
        
        List<ChunkBoundary> boundaries = determineBoundaries(cachedContent, cachedTokens);
        float[][] chunkEmbeddings = new float[boundaries.size()][];
        
        for (int i = 0; i < boundaries.size(); i++) {
            ChunkBoundary boundary = boundaries.get(i);
            chunkEmbeddings[i] = poolTokenEmbeddings(cachedTokenEmbeddings, 
                                                     boundary.startToken, 
                                                     boundary.endToken);
        }
        
        return chunkEmbeddings;
    }
    
    /**
     * Determines chunk boundaries based on the configured boundary type.
     */
    private List<ChunkBoundary> determineBoundaries(String content, String[] tokens) {
        List<ChunkBoundary> boundaries = new ArrayList<>();
        
        switch (boundaryType) {
            case SENTENCE:
                boundaries = findSentenceBoundaries(content, tokens);
                break;
            case PARAGRAPH:
                boundaries = findParagraphBoundaries(content, tokens);
                break;
            case FIXED_TOKENS:
                boundaries = findFixedTokenBoundaries(tokens);
                break;
        }
        
        return boundaries;
    }
    
    /**
     * Finds sentence boundaries in the token sequence.
     */
    private List<ChunkBoundary> findSentenceBoundaries(String content, String[] tokens) {
        List<ChunkBoundary> boundaries = new ArrayList<>();
        Pattern sentenceEnd = Pattern.compile("[.!?]+\\s+");
        Matcher matcher = sentenceEnd.matcher(content);
        
        int lastCharEnd = 0;
        int lastTokenEnd = 0;
        
        while (matcher.find()) {
            int charEnd = matcher.end();
            int tokenEnd = findTokenIndexForCharPosition(content, tokens, charEnd);
            
            if (tokenEnd > lastTokenEnd) {
                boundaries.add(new ChunkBoundary(lastTokenEnd, tokenEnd, lastCharEnd, charEnd));
                lastTokenEnd = tokenEnd;
                lastCharEnd = charEnd;
            }
        }
        
        // Add final chunk
        if (lastTokenEnd < tokens.length) {
            boundaries.add(new ChunkBoundary(lastTokenEnd, tokens.length, 
                                            lastCharEnd, content.length()));
        }
        
        return boundaries;
    }
    
    /**
     * Finds paragraph boundaries in the token sequence.
     */
    private List<ChunkBoundary> findParagraphBoundaries(String content, String[] tokens) {
        List<ChunkBoundary> boundaries = new ArrayList<>();
        Pattern paragraphEnd = Pattern.compile("\\n\\s*\\n");
        Matcher matcher = paragraphEnd.matcher(content);
        
        int lastCharEnd = 0;
        int lastTokenEnd = 0;
        
        while (matcher.find()) {
            int charEnd = matcher.end();
            int tokenEnd = findTokenIndexForCharPosition(content, tokens, charEnd);
            
            if (tokenEnd > lastTokenEnd) {
                boundaries.add(new ChunkBoundary(lastTokenEnd, tokenEnd, lastCharEnd, charEnd));
                lastTokenEnd = tokenEnd;
                lastCharEnd = charEnd;
            }
        }
        
        // Add final chunk
        if (lastTokenEnd < tokens.length) {
            boundaries.add(new ChunkBoundary(lastTokenEnd, tokens.length, 
                                            lastCharEnd, content.length()));
        }
        
        return boundaries;
    }
    
    /**
     * Finds fixed-token-count boundaries.
     */
    private List<ChunkBoundary> findFixedTokenBoundaries(String[] tokens) {
        List<ChunkBoundary> boundaries = new ArrayList<>();
        
        for (int i = 0; i < tokens.length; i += tokensPerChunk) {
            int endToken = Math.min(i + tokensPerChunk, tokens.length);
            boundaries.add(new ChunkBoundary(i, endToken, -1, -1)); // Char positions not needed
        }
        
        return boundaries;
    }
    
    /**
     * Finds the token index corresponding to a character position in the text.
     */
    private int findTokenIndexForCharPosition(String content, String[] tokens, int charPos) {
        int currentCharPos = 0;
        
        for (int i = 0; i < tokens.length; i++) {
            currentCharPos += tokens[i].length();
            if (currentCharPos >= charPos) {
                return i + 1;
            }
        }
        
        return tokens.length;
    }
    
    /**
     * Extracts the text for a chunk based on boundaries.
     */
    private String extractChunkText(String content, String[] tokens, ChunkBoundary boundary) {
        if (boundary.startChar >= 0 && boundary.endChar >= 0) {
            // Use character positions if available
            return content.substring(boundary.startChar, 
                                   Math.min(boundary.endChar, content.length()));
        } else {
            // Reconstruct from tokens
            StringBuilder sb = new StringBuilder();
            for (int i = boundary.startToken; i < boundary.endToken && i < tokens.length; i++) {
                sb.append(tokens[i]);
                if (i < boundary.endToken - 1) {
                    sb.append(" ");
                }
            }
            return sb.toString();
        }
    }
    
    /**
     * Pools token embeddings within a range using the configured pooling strategy.
     */
    private float[] poolTokenEmbeddings(float[][] tokenEmbeddings, int startToken, int endToken) {
        int embeddingDim = tokenEmbeddings[0].length;
        float[] pooled = new float[embeddingDim];
        
        switch (poolingStrategy) {
            case MEAN:
                // Average all token embeddings
                for (int i = startToken; i < endToken; i++) {
                    for (int j = 0; j < embeddingDim; j++) {
                        pooled[j] += tokenEmbeddings[i][j];
                    }
                }
                int count = endToken - startToken;
                for (int j = 0; j < embeddingDim; j++) {
                    pooled[j] /= count;
                }
                break;
                
            case MAX:
                // Take max value for each dimension
                for (int j = 0; j < embeddingDim; j++) {
                    float max = Float.NEGATIVE_INFINITY;
                    for (int i = startToken; i < endToken; i++) {
                        max = Math.max(max, tokenEmbeddings[i][j]);
                    }
                    pooled[j] = max;
                }
                break;
                
            case FIRST:
                // Use first token's embedding
                System.arraycopy(tokenEmbeddings[startToken], 0, pooled, 0, embeddingDim);
                break;
        }
        
        return pooled;
    }
    
    @Override
    public String getName() {
        return "Late Chunking";
    }
    
    @Override
    public String getDescription() {
        return "Embeds entire document first to preserve full context, then applies chunk boundaries. " +
               "Each chunk retains semantic understanding from the complete document, improving " +
               "retrieval accuracy for queries with long-range dependencies. " +
               "Boundary: " + boundaryType + ", Pooling: " + poolingStrategy;
    }
    
    /**
     * Internal class to represent chunk boundaries.
     */
    private static class ChunkBoundary {
        final int startToken;
        final int endToken;
        final int startChar;
        final int endChar;
        
        ChunkBoundary(int startToken, int endToken, int startChar, int endChar) {
            this.startToken = startToken;
            this.endToken = endToken;
            this.startChar = startChar;
            this.endChar = endChar;
        }
    }
}

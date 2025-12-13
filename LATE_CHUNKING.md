# Late Chunking Strategy

## Overview

**Late Chunking** is an advanced method for preparing long documents for retrieval systems, designed to overcome the critical problem of context loss that occurs in traditional document processing.

## The Problem with Traditional Chunking

Traditional RAG systems follow this approach:
1. **Split first**: Break document into small chunks
2. **Embed second**: Generate embeddings for each isolated chunk
3. **Result**: Each chunk loses context from other parts of the document

This causes:
- Loss of long-range dependencies
- Poor performance on queries requiring cross-chunk understanding
- Inability to capture document-level semantics

## How Late Chunking Works

Late Chunking **reverses the order**:

### Step 1: Embed First (Full Context)
```
Document → Token-Level Embeddings (entire document)
```
Uses a long-context embedding model to process the entire document (or largest possible segment) in a single pass. Each token's embedding vector is infused with **global, full-document context** and semantic relationships from every other part of the text.

### Step 2: Chunk Later (Preserve Context)
```
Token Embeddings → Apply Boundaries → Pool by Chunk → Context-Rich Chunk Embeddings
```
After embedding, the system applies pre-defined segment boundaries (sentences, paragraphs, or fixed token spans) to the token-level embeddings. For each segment, pooling (typically mean pooling) combines token vectors into a single chunk embedding.

### Result
Smaller, retrievable chunk embeddings that:
- ✅ Have compact size (efficient retrieval)
- ✅ Retain deep, accurate semantic context (full document awareness)
- ✅ Improve accuracy for long-range dependencies
- ✅ Better handle cross-chunk references

## Architecture

### TokenLevelEmbeddingProvider Interface
```java
public interface TokenLevelEmbeddingProvider extends EmbeddingProvider {
    // Generate token-level embeddings for entire document
    float[][] embedTokenLevel(String text);
    
    // Get token sequence matching embeddings
    String[] tokenize(String text);
    
    // Get max context length
    int getMaxContextLength();
}
```

### LateChunking Class
```java
public class LateChunking implements ChunkingStrategy {
    // Boundary types: SENTENCE, PARAGRAPH, FIXED_TOKENS
    // Pooling strategies: MEAN, MAX, FIRST
}
```

## Usage

### Basic Usage (Sentence Boundaries, Mean Pooling)

```java
// Create a token-level embedding provider
TokenLevelEmbeddingProvider provider = new YourTokenLevelProvider();

// Create Late Chunking strategy
ChunkingStrategy strategy = new LateChunking(provider);

// Use with RAGService
List<String> chunks = strategy.chunk(documentContent);

// Get context-rich embeddings for each chunk
LateChunking lateChunking = (LateChunking) strategy;
float[][] chunkEmbeddings = lateChunking.getAllChunkEmbeddings();
```

### Advanced Usage (Custom Settings)

```java
// Paragraph boundaries with max pooling
ChunkingStrategy strategy = new LateChunking(
    provider,
    LateChunking.BoundaryType.PARAGRAPH,
    128,  // tokens per chunk (for FIXED_TOKENS mode)
    LateChunking.PoolingStrategy.MAX
);

// Fixed token count with mean pooling
ChunkingStrategy fixedStrategy = new LateChunking(
    provider,
    LateChunking.BoundaryType.FIXED_TOKENS,
    256,  // 256 tokens per chunk
    LateChunking.PoolingStrategy.MEAN
);
```

### Integration with RAGService

```java
try (RAGService rag = new RAGService(indexPath, provider)) {
    LateChunking strategy = new LateChunking(
        (TokenLevelEmbeddingProvider) provider,
        LateChunking.BoundaryType.SENTENCE,
        128,
        LateChunking.PoolingStrategy.MEAN
    );
    
    // Chunk the document
    List<String> chunks = strategy.chunk(documentContent);
    
    // Get context-rich embeddings
    float[][] embeddings = strategy.getAllChunkEmbeddings();
    
    // Index with custom embeddings
    for (int i = 0; i < chunks.size(); i++) {
        String chunkId = "doc_chunk_" + i;
        // Use custom embedding instead of re-embedding the chunk
        rag.addDocumentWithEmbedding(chunkId, chunks.get(i), embeddings[i]);
    }
    
    rag.commit();
}
```

## Configuration Options

### Boundary Types

| Type | Description | Best For |
|------|-------------|----------|
| `SENTENCE` | Split on sentence endings (`.`, `!`, `?`) | Natural language documents, articles |
| `PARAGRAPH` | Split on paragraph breaks (double newlines) | Structured documents, blog posts |
| `FIXED_TOKENS` | Split every N tokens | Code, logs, uniform chunking needs |

### Pooling Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `MEAN` | Average all token embeddings | Most use cases (balanced representation) |
| `MAX` | Take maximum value per dimension | Preserving salient features |
| `FIRST` | Use first token only | BERT-style CLS token approaches |

## Performance Considerations

### Memory Usage
- **Higher than traditional chunking**: Stores all token-level embeddings in memory
- **Mitigation**: Process very long documents in sections

### Computation
- **Single embedding pass**: More efficient than embedding many small chunks
- **Trade-off**: Requires long-context embedding model

### Best Results
Works best when:
- Document has cross-references
- Queries require understanding of document-level context
- Long-range dependencies matter (e.g., pronouns referring to earlier entities)

## Example: Research Paper

Traditional chunking loses context:
```
Chunk 1: "The experiment showed significant results..."
Chunk 2: "These findings contradict earlier work..." ❌ What findings?
```

Late Chunking preserves context:
```
Chunk 1: [Embedding knows full paper context]
Chunk 2: [Embedding knows "these findings" = results from Chunk 1] ✅
```

## Implementing TokenLevelEmbeddingProvider

To use Late Chunking, implement the `TokenLevelEmbeddingProvider` interface:

```java
public class MyTokenLevelProvider implements TokenLevelEmbeddingProvider {
    
    @Override
    public float[][] embedTokenLevel(String text) {
        // Call your embedding model's token-level API
        // Example: OpenAI with output_dimensions or similar
        // Return: [numTokens][embeddingDim]
    }
    
    @Override
    public String[] tokenize(String text) {
        // Tokenize text matching your embedding model
        // Must align with embedTokenLevel() output
    }
    
    @Override
    public int getMaxContextLength() {
        // Return model's max context (e.g., 8192 for GPT models)
        return 8192;
    }
    
    @Override
    public float[] embed(String text) {
        // Regular embedding (average of token embeddings)
        float[][] tokenEmbeddings = embedTokenLevel(text);
        return meanPool(tokenEmbeddings);
    }
    
    @Override
    public int getDimension() {
        return 1536; // Your embedding dimension
    }
}
```

## Comparison

| Aspect | Traditional Chunking | Late Chunking |
|--------|---------------------|---------------|
| **Process** | Split → Embed | Embed → Split |
| **Context** | Per-chunk only | Full document |
| **Accuracy** | Lower for cross-chunk queries | Higher for all queries |
| **Memory** | Lower | Higher (stores token embeddings) |
| **Speed** | Slower (many embed calls) | Faster (one embed call) |
| **Use Case** | Simple retrieval | Context-critical retrieval |

## When to Use Late Chunking

✅ **Use Late Chunking when:**
- Document has internal cross-references
- Queries need document-level understanding
- Pronouns/anaphora resolution is important
- Technical/legal documents with dependencies
- Research papers with result discussions

❌ **Skip Late Chunking when:**
- Documents are independent fragments
- Simple keyword matching suffices
- Memory is severely constrained
- Embedding provider doesn't support token-level output

## Future Enhancements

Potential improvements:
- Adaptive pooling (attention-weighted instead of mean)
- Hierarchical chunking (embed at multiple levels)
- Sliding window with overlap at token level
- Dynamic boundary detection using embedding similarity

---

**Package:** `io.github.vishalmysore.rag.chunking.LateChunking`

**Requires:** `TokenLevelEmbeddingProvider` implementation

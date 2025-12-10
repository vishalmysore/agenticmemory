# Chunking Strategies Framework

The LuceneRAG library includes an extensible chunking framework that allows you to split documents into optimal chunks for semantic search and retrieval.

## Architecture

All chunking strategies are part of the **core framework** in the `io.github.vishalmysore.chunking` package. The example code simply demonstrates how to use these strategies.

### Core Components

1. **`ChunkingStrategy` interface** - Base interface for all chunking strategies
   - `List<String> chunk(String content)` - Splits content into chunks
   - `String getName()` - Returns strategy name
   - `String getDescription()` - Returns strategy description

2. **`RAGService.addDocumentWithChunking()`** - Convenience method for automatic chunking
   ```java
   int chunkCount = rag.addDocumentWithChunking(
       "document_id", 
       content, 
       chunkingStrategy
   );
   ```

## Built-in Strategies

### 1. Sliding Window Chunking
**Package:** `io.github.vishalmysore.chunking.SlidingWindowChunking`

Creates overlapping chunks to preserve context across boundaries.

```java
ChunkingStrategy strategy = new SlidingWindowChunking(150, 30);
// 150 words per chunk, 30 words overlap (20%)
```

**Best for:** Healthcare records, continuous narratives, patient notes

---

### 2. Adaptive Chunking
**Package:** `io.github.vishalmysore.chunking.AdaptiveChunking`

Respects natural document boundaries while staying within token limits.

```java
ChunkingStrategy strategy = new AdaptiveChunking(
    "(?m)^SECTION \\d+:",  // Boundary pattern
    800,                    // Min tokens
    1200                    // Max tokens
);
```

**Best for:** Legal contracts, structured documents, policy documents

---

### 3. Entity-Based Chunking
**Package:** `io.github.vishalmysore.chunking.EntityBasedChunking`

Groups sentences by mentioned entities (people, companies, locations).

```java
String[] entities = {"Elon Musk", "Tesla", "SpaceX"};
ChunkingStrategy strategy = new EntityBasedChunking(entities);
```

**Best for:** News articles, research papers, multi-person biographies

---

### 4. Topic/Theme-Based Chunking
**Package:** `io.github.vishalmysore.chunking.TopicBasedChunking`

Groups content by underlying topics or themes.

```java
ChunkingStrategy strategy = new TopicBasedChunking(
    "(EDUCATION|CAREER|PATENTS):"
);
```

**Best for:** Research papers, technical documentation, structured content

---

### 5. Hybrid Chunking
**Package:** `io.github.vishalmysore.chunking.HybridChunking`

Combines multiple strategies in a pipeline.

```java
ChunkingStrategy adaptive = new AdaptiveChunking("(?m)^===\\s*$");
ChunkingStrategy topic = new TopicBasedChunking("(INTRO|BODY|CONCLUSION):");

ChunkingStrategy strategy = new HybridChunking(adaptive, topic);
```

**Best for:** Complex documents requiring multi-stage processing

---

### 6. Task-Aware Chunking
**Package:** `io.github.vishalmysore.chunking.TaskAwareChunking`

Adapts chunking based on downstream task (summarization/search/Q&A).

```java
// For summarization (small chunks)
ChunkingStrategy strategy = new TaskAwareChunking(TaskType.SUMMARIZATION);

// For search (medium chunks - signatures + docstrings)
ChunkingStrategy strategy = new TaskAwareChunking(TaskType.SEARCH);

// For Q&A (large chunks - full context)
ChunkingStrategy strategy = new TaskAwareChunking(TaskType.QA);
```

**Best for:** Code repositories, multi-purpose document processing

---

### 7. HTML Tag-Based Chunking
**Package:** `io.github.vishalmysore.chunking.HTMLTagBasedChunking`

Uses HTML/XML structure to determine boundaries.

```java
ChunkingStrategy strategy = new HTMLTagBasedChunking(
    "h2",   // Split on <h2> tags
    true    // Strip HTML tags from output
);
```

**Best for:** Web content, HTML documentation, XML documents

---

### 8. Code-Specific Chunking
**Package:** `io.github.vishalmysore.chunking.CodeSpecificChunking`

Uses language syntax for AST-inspired boundaries.

```java
ChunkingStrategy strategy = new CodeSpecificChunking(
    CodeSpecificChunking.Language.PYTHON
);
// Also supports JAVA and JAVASCRIPT
```

**Best for:** Source code, code repositories, programming tutorials

---

### 9. Regex Chunking
**Package:** `io.github.vishalmysore.chunking.RegexChunking`

Uses regex patterns to identify boundaries and group chunks.

```java
ChunkingStrategy strategy = new RegexChunking(
    "\\[\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\]",  // Split pattern
    "(ERROR|WARN|INFO|DEBUG)"                            // Group pattern
);
```

**Best for:** Log files, structured text, timestamp-based data

---

## Creating Custom Strategies

The framework is **fully extensible**. Create your own chunking strategy by implementing `ChunkingStrategy`:

```java
public class MyCustomChunking implements ChunkingStrategy {
    
    @Override
    public List<String> chunk(String content) {
        // Your chunking logic here
        List<String> chunks = new ArrayList<>();
        // ... split content into chunks ...
        return chunks;
    }
    
    @Override
    public String getName() {
        return "My Custom Chunking";
    }
    
    @Override
    public String getDescription() {
        return "Splits content using my custom algorithm";
    }
}
```

Then use it with `RAGService`:

```java
ChunkingStrategy myStrategy = new MyCustomChunking();
rag.addDocumentWithChunking("doc_id", content, myStrategy);
```

See `CustomChunkingExample.java` for a complete example.

## Usage Examples

### Basic Usage

```java
// 1. Create a chunking strategy
ChunkingStrategy strategy = new SlidingWindowChunking(150, 30);

// 2. Use it with RAGService
try (RAGService rag = new RAGService(indexPath, embeddingProvider)) {
    int chunkCount = rag.addDocumentWithChunking(
        "my_document",
        documentContent,
        strategy
    );
    rag.commit();
    
    System.out.println("Created " + chunkCount + " chunks");
}
```

### Manual Chunking

If you prefer to manually control chunk IDs:

```java
ChunkingStrategy strategy = new EntityBasedChunking("Entity1", "Entity2");
List<String> chunks = strategy.chunk(content);

for (int i = 0; i < chunks.size(); i++) {
    String customId = "custom_prefix_" + i;
    rag.addDocument(customId, chunks.get(i));
}
```

## Running the Examples

### Comprehensive Demo (All 9 Strategies)

```bash
export OPENAI_API_KEY="your-key-here"
mvn exec:java -Dexec.mainClass="io.github.vishalmysore.rag.examples.ChunkingStrategiesExample" -q
```

### Custom Strategy Example

```bash
export OPENAI_API_KEY="your-key-here"
mvn exec:java -Dexec.mainClass="io.github.vishalmysore.rag.examples.CustomChunkingExample" -q
```

## Design Principles

1. **Core Framework Integration** - All strategies are part of the core library, not examples
2. **Extensibility** - Simple interface allows custom implementations
3. **Composability** - Strategies can be combined (see `HybridChunking`)
4. **Task-Specific** - Different strategies for different use cases
5. **Zero Configuration** - Sensible defaults with optional customization

## When to Use Which Strategy

| Use Case | Recommended Strategy |
|----------|---------------------|
| Healthcare records, patient notes | Sliding Window |
| Legal contracts, policies | Adaptive |
| News articles, multi-entity docs | Entity-Based |
| Research papers, structured content | Topic-Based |
| Complex multi-format docs | Hybrid |
| Code repositories | Code-Specific or Task-Aware |
| Web pages, HTML docs | HTML Tag-Based |
| Server logs, timestamped data | Regex |

## API Reference

### ChunkingStrategy Interface

```java
public interface ChunkingStrategy {
    List<String> chunk(String content);
    String getName();
    String getDescription();
}
```

### RAGService Methods

```java
// Add document with automatic chunking
int addDocumentWithChunking(String baseId, String content, ChunkingStrategy strategy)

// Add document with chunking and metadata
int addDocumentWithChunking(String baseId, String content, ChunkingStrategy strategy, 
                           Map<String, String> metadata)
```

## License

Same as LuceneRAG library.

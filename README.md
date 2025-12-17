# Agentic Memory Foundation

A **lightweight, embedded RAG (Retrieval-Augmented Generation) library** supporting Zettelkasten method built on Apache Lucene that provides the foundational infrastructure for building **agentic memory systems**. Store and search documents using vector embeddings with advanced chunking strategies—**no external server required**.

## What is Agentic Memory?

This library provides the core RAG infrastructure needed to build agentic memory systems—AI systems that autonomously organize, link, and evolve their knowledge over time. While the full vision of agentic memory includes LLM-driven note construction, autonomous link generation, and dynamic memory evolution, this library delivers the essential foundation: **flexible document storage, advanced chunking strategies, and powerful retrieval mechanisms**.

```
   <dependency>
            <groupId>io.github.vishalmysore</groupId>
            <artifactId>agenticmemory</artifactId>
            <version>0.1.1</version>
        </dependency>
```

## Features

✅ **No External Server Required** - All data stored locally in Lucene index files  
✅ **Vector Similarity Search** - Using Lucene's KnnFloatVectorField for semantic search  
✅ **Keyword Search** - Traditional full-text search capabilities  
✅ **Hybrid Search** - Combine vector and keyword search with custom weights  
✅ **Metadata Support** - Attach custom metadata to documents  
✅ **Advanced Chunking Strategies** - 10+ extensible chunking strategies for optimal document segmentation  
✅ **Extensible Architecture** - Plugin your own chunking strategies and embedding providers  
✅ **Simple API** - Easy to use, clean interface  
✅ **Fully Tested** - Comprehensive unit tests included  

## Why "Agentic Memory Foundation"?

This library is positioned as a **foundation** for agentic memory because it provides:

1. **Flexible Knowledge Representation** - Multiple chunking strategies (sliding window, entity-based, topic-based, NER-based, etc.) allow adaptive knowledge organization
2. **Metadata-Rich Storage** - Documents can carry rich contextual metadata, enabling future link generation and relationship mapping
3. **Hybrid Retrieval** - Combines vector similarity with keyword search for context-aware retrieval
4. **Extensible Design** - ChunkingStrategy interface and EmbeddingProvider interface allow custom implementations

**What's Missing for Full Agentic Memory?**
- LLM-driven autonomous note construction
- Automatic link generation between related concepts
- Dynamic memory evolution and consolidation
- Graph-based context retrieval with relationship traversal




## Storage Architecture

This library uses **local file-based storage** via Apache Lucene:
- **No Qdrant server needed** - Unlike some vector databases, this runs entirely locally
- **No Docker containers** - No external dependencies to manage
- **File-based index** - Lucene creates and manages index files on your local filesystem
- **Embedded solution** - Everything runs in your Java application process



## Advanced Chunking Strategies

One of the key features that makes this library suitable as an agentic memory foundation is its **extensible chunking system**. Documents can be broken into chunks using various strategies before indexing, enabling more precise retrieval and knowledge organization.

### Available Chunking Strategies

| Strategy | Use Case | Example |
|----------|----------|---------|
| **SlidingWindowChunking** | General text processing | Research papers, articles |
| **AdaptiveChunking** | Section-based documents | Markdown, structured content |
| **EntityBasedChunking** | Entity-focused retrieval | Biographical data, company info |
| **TopicBasedChunking** | Topic segmentation | Multi-topic documents |
| **HybridChunking** | Complex multi-stage processing | Technical documentation |
| **TaskAwareChunking** | Task-specific chunking | Different chunking per use case |
| **HTMLTagBasedChunking** | Web content preservation | HTML documents, web scraping |
| **CodeSpecificChunking** | Source code files | Python, Java, etc. |
| **RegexChunking** | Pattern-based splitting | Log files, structured data |
| **NERBasedChunking** | Automatic entity detection | ML-powered entity extraction |

### Using Chunking Strategies

```java
import io.github.vishalmysore.*;
import io.github.vishalmysore.chunking.*;

// Create a chunking strategy
ChunkingStrategy strategy = new SlidingWindowChunking(150, 30); // 150 words, 30 overlap

// Add document with chunking
try (RAGService rag = new RAGService(indexPath, embeddings)) {
    rag.addDocumentWithChunking("doc1", longDocument, strategy);
    rag.commit();
    
    // Each chunk is indexed separately with ID: doc1_chunk_0, doc1_chunk_1, etc.
}
```

### Creating Custom Chunking Strategies

Implement the `ChunkingStrategy` interface:

```java
public class MyCustomChunking implements ChunkingStrategy {
    @Override
    public List<String> chunk(String content) {
        // Your custom logic here
        return Arrays.asList(content.split("CUSTOM_DELIMITER"));
    }
    
    @Override
    public String getName() {
        return "Custom Chunking";
    }
    
    @Override
    public String getDescription() {
        return "My custom chunking logic";
    }
}
```

See `examples/ChunkingStrategiesExample.java` and `CHUNKING_STRATEGIES.md` for detailed documentation.

## Quick Start

### 1. Basic Usage

```java
import io.github.vishalmysore.*;
import java.nio.file.Paths;

// Create a local index directory (stored on filesystem)
Path indexPath = Paths.get("my-rag-index");

// Initialize with embedding provider
EmbeddingProvider embeddings = new MockEmbeddingProvider(128);

try (RAGService rag = new RAGService(indexPath, embeddings)) {
    // Add documents
    rag.addDocument("doc1", "Machine learning is a subset of AI");
    rag.addDocument("doc2", "Python is a programming language");
    rag.commit();
    
    // Search semantically
    List<SearchResult> results = rag.search("artificial intelligence", 5);
    
    for (SearchResult result : results) {
        System.out.println(result.getContent() + " - Score: " + result.getScore());
    }
}
```

### 2. With Metadata

```java
Map<String, String> metadata = new HashMap<>();
metadata.put("author", "John Doe");
metadata.put("date", "2025-12-09");

rag.addDocument("doc1", "Content here", metadata);
rag.commit();

// Search returns metadata
List<SearchResult> results = rag.search("query", 10);
String author = results.get(0).getMetadata("author");
```

### 3. Retrieve Context for RAG

```java
// Get combined context from top-k documents
String context = rag.retrieveContext("What is machine learning?", 3);

// Use this context with your LLM
String prompt = "Context: " + context + "\n\nQuestion: What is machine learning?";
```

### 4. Load Previously Indexed Documents

The library persists all data to disk, so you can reload documents later:

```java
// Index documents in one session
try (RAGService rag = new RAGService(indexPath, embeddings)) {
    rag.addDocument("doc1", "Machine learning content");
    rag.addDocument("doc2", "AI content");
    rag.commit();
}

// Later, load the index in a new session
try (RAGService rag = new RAGService(indexPath, embeddings)) {
    // Get total document count
    int count = rag.getDocumentCount();
    System.out.println("Found " + count + " documents");
    
    // Retrieve a specific document by ID
    Document doc = rag.getDocumentById("doc1");
    if (doc != null) {
        System.out.println("Content: " + doc.getContent());
        System.out.println("Author: " + doc.getMetadata("author"));
    }
    
    // Retrieve all documents
    List<Document> allDocs = rag.getAllDocuments();
    for (Document d : allDocs) {
        System.out.println(d.getId() + ": " + d.getContent());
    }
    
    // Check if a document exists
    boolean exists = rag.documentExists("doc1");
    
    // Continue adding more documents to existing index
    rag.addDocument("doc3", "New content");
    rag.commit();
}
```

### 5. Low-Level API

```java
// Direct access to Lucene engine
LuceneRAGEngine engine = new LuceneRAGEngine(indexPath, 128);

// Create document with vector
Document doc = new Document.Builder()
    .id("doc1")
    .content("Text content")
    .vector(new float[]{0.1f, 0.2f, ...})
    .addMetadata("key", "value")
    .build();

engine.indexDocument(doc);
engine.commit();

// Vector search
float[] queryVector = {0.1f, 0.2f, ...};
List<SearchResult> results = engine.vectorSearch(queryVector, 10);

// Hybrid search (combines vector + keyword)
List<SearchResult> hybrid = engine.hybridSearch(
    queryVector, 
    "keyword query", 
    10,
    0.7f  // 70% vector, 30% keyword
);

engine.close();
```

## API Overview

### RAGService (High-Level API)

| Method | Description |
|--------|-------------|
| `addDocument(id, content)` | Add document with auto-generated embedding |
| `addDocument(id, content, metadata)` | Add document with metadata |
| `search(query, topK)` | Semantic search using embeddings |
| `keywordSearch(query, topK)` | Traditional keyword search |
| `hybridSearch(query, topK, weight)` | Combined vector + keyword search |
| `retrieveContext(query, topK)` | Get concatenated context for RAG |
| `getDocumentById(id)` | Retrieve a specific document by ID |
| `getAllDocuments()` | Retrieve all indexed documents |
| `documentExists(id)` | Check if a document exists |
| `getDocumentCount()` | Get total number of documents |
| `deleteDocument(id)` | Remove a document |
| `commit()` | Persist changes to disk |

### LuceneRAGEngine (Low-Level API)

| Method | Description |
|--------|-------------|
| `indexDocument(document)` | Index a document with vector |
| `indexDocuments(documents)` | Batch index multiple documents |
| `vectorSearch(vector, topK)` | KNN vector similarity search |
| `keywordSearch(text, topK)` | Full-text keyword search |
| `hybridSearch(...)` | Weighted combination search |
| `getDocumentById(id)` | Retrieve a document by ID |
| `getAllDocuments()` | Get all documents from index |
| `documentExists(id)` | Check document existence |
| `deleteDocument(id)` | Delete by ID |
| `getDocumentCount()` | Count indexed documents |

## Embedding Providers

The library supports multiple embedding providers:

### MockEmbeddingProvider (For Testing)

Simple hash-based embeddings for testing and development:

```java
EmbeddingProvider embeddings = new MockEmbeddingProvider(128);
```

### OpenAIEmbeddingProvider (Production)

Use OpenAI's embedding models for production:

```java
// With custom API URL
OpenAIEmbeddingProvider embeddings = new OpenAIEmbeddingProvider(
    "https://api.openai.com/v1/embeddings",  // API URL
    "your-api-key",                           // API Key
    "text-embedding-3-small",                 // Model
    1536                                      // Dimension
);

// Or use default OpenAI endpoint
OpenAIEmbeddingProvider embeddings = new OpenAIEmbeddingProvider(
    "your-api-key",
    "text-embedding-3-small",
    1536
);

// Supported models:
// - text-embedding-3-small (dimensions: 512, 1536)
// - text-embedding-3-large (dimensions: 256, 1024, 3072)
// - text-embedding-ada-002 (dimensions: 1536)

// Use with RAGService
try (RAGService rag = new RAGService(indexPath, embeddings)) {
    rag.addDocument("doc1", "Your content here");
    rag.commit();
}

// Don't forget to close when done
embeddings.close();
```

### Custom Embedding Providers

Implement the `EmbeddingProvider` interface for other models:

```java
public interface EmbeddingProvider {
    float[] embed(String text);
    int getDimension();
}
```

Examples:
- **Sentence Transformers** - Local models via ONNX or Python bridge
- **Azure OpenAI** - Use Azure's OpenAI service
- **Custom Models** - Any embedding model you prefer

## How It Works

1. **Indexing**: Documents are converted to vectors (embeddings) and stored in a local Lucene index
2. **Storage**: Lucene creates index files in the specified directory on your filesystem
3. **Search**: Query vectors are compared against stored vectors using cosine similarity (KNN)
4. **Results**: Top-K most similar documents are returned with scores

## Index Storage Location

The index is stored as **local files** in the directory you specify:

```java
Path indexPath = Paths.get("./lucene-index");  // Creates files in ./lucene-index/
```

You'll see files like:
- `segments_*`
- `*.fdx`, `*.fdt` (stored fields)
- `*.fnm` (field names)
- `*.nvd`, `*.nvm` (vector data)

**No server process runs** - the index is just files that Lucene reads/writes.






## Why This Architecture?

### Current Strengths (RAG Foundation)
- **Local-first**: No external dependencies, pure file-based storage
- **Flexible**: Multiple chunking strategies for different content types
- **Extensible**: Plugin architecture for custom strategies and providers
- **Efficient**: Lucene's HNSW algorithm for fast vector search
- **Metadata-rich**: Supports arbitrary key-value annotations

### Design Decisions for Agentic Memory
- **Separation of Concerns**: Core RAG (`io.github.vishalmysore.rag`) vs. Agentic features (`io.github.vishalmysore.agenticmemory`)
- **Interface-driven**: `ChunkingStrategy` and `EmbeddingProvider` interfaces enable extensibility
- **Metadata Foundation**: Rich metadata support enables future link tracking and relationship mapping
- **Chunking Flexibility**: Different chunking strategies lay groundwork for task-aware knowledge organization


---

## 🧠 Zettelkasten-Style Knowledge Management

Library includes  complete **Zettelkasten-style linking system** for advanced knowledge management, built on top of the agentic memory foundation. This system enables atomic note creation, automatic link generation, and graph-based knowledge navigation.

### Key Features

✅ **Atomic Notes (ZettelNote)** - Single-concept notes with bidirectional links  
✅ **Automatic Link Generation** - Entity, semantic, and tag-based connections  
✅ **Graph Algorithms** - BFS/DFS traversal, shortest paths, connected components  
✅ **Link-Aware Retrieval** - Context expansion via graph traversal  
✅ **Entity Extraction** - OpenNLP + pattern-based entity detection  
✅ **Atomic Chunking** - Splits content into linkable atomic ideas (300-500 words)  
✅ **Link Storage** - Persistent link management in RAG index with caching  
✅ **Graph Analytics** - Centrality, bridge detection, cluster analysis  

### Core Components

#### 1. **ZettelNote** - Enhanced Note Structure
```java
ZettelNote note = new ZettelNote.Builder()
    .id("note-1")
    .content("Machine learning is a subset of AI...")
    .summary("ML overview")
    .tags(Arrays.asList("machine-learning", "ai"))
    .entities(Arrays.asList("Machine Learning", "AI"))
    .build();

// Atomicity validation
if (note.isAtomic()) {
    System.out.println("Note focuses on a single concept");
}

// Connectivity metrics
double score = note.getConnectivityScore(); // link count × avg strength
```

#### 2. **ZettelkastenRAGService** - Main API
```java
// Initialize with index path and embeddings
ZettelkastenRAGService zettelRAG = new ZettelkastenRAGService(
    "zettelkasten-index",
    new MockEmbeddingProvider(128)
);

// Create atomic notes with auto-linking
ZettelNote note1 = zettelRAG.createNote(
    "Neural networks consist of layers of interconnected nodes...",
    Arrays.asList("neural-networks", "deep-learning")
);

// Links are automatically generated based on:
// - Entity co-occurrence
// - Semantic similarity (embeddings)
// - Tag overlap

// Explore auto-generated links
for (Link link : note1.getOutgoingLinks()) {
    System.out.printf("%s -> %s [%s, strength: %.2f]\n",
        link.getSourceNoteId(),
        link.getTargetNoteId(),
        link.getType(),
        link.getStrength()
    );
}
```

#### 3. **Graph-Based Retrieval**
```java
// Traditional RAG: Top-K direct matches
List<SearchResult> direct = zettelRAG.search("neural networks", 3);

// Zettelkasten RAG: Context-expanded retrieval
String context = zettelRAG.retrieveContextWithLinks(
    "neural networks",
    topK: 3,
    linkDepth: 2  // Include notes within 2 hops
);

// Context now includes directly matching notes + linked notes
// Example: "neural networks" query also retrieves:
// - Notes about "backpropagation" (linked via ELABORATES)
// - Notes about "deep learning" (linked via SIMILAR_TOPIC)
// - Notes about "activation functions" (linked via REFERENCES)
```

#### 4. **Graph Traversal & Analytics**
```java
ZettelkastenGraph graph = zettelRAG.getGraph();

// BFS traversal from a note (explore knowledge neighborhood)
List<ZettelNote> neighborhood = graph.traverseFromNote("note-1", maxDepth: 2);

// Find shortest path between concepts
List<ZettelNote> path = graph.findPath("neural-networks-note", "backprop-note");

// Identify knowledge hubs (most connected notes)
List<ZettelNote> hubs = graph.getMostConnectedNotes(topN: 5);

// Detect knowledge clusters
List<List<String>> clusters = graph.findConnectedComponents();

// Find critical bridge notes (connecting different clusters)
List<String> bridges = graph.findBridgeNotes();
```

#### 5. **Link Types**
The system automatically assigns link types based on relationship characteristics:

| Link Type | Description | Example |
|-----------|-------------|---------|
| `REFERENCES` | High semantic similarity (>0.7) | "Neural networks" ↔ "Deep learning" |
| `ELABORATES` | Detailed expansion of concept | "ML overview" ↔ "Gradient descent explained" |
| `CONTRADICTS` | Opposing viewpoints | "Frequentist stats" ↔ "Bayesian approach" |
| `SIMILAR_TOPIC` | Tag overlap | Notes tagged "machine-learning" |
| `RELATED_ENTITY` | Entity co-occurrence | Both mention "TensorFlow" |
| `CAUSES` | Causal relationship | "Overfitting" ↔ "Small dataset" |

#### 6. **Entity-Based Search**
```java
// Find notes containing specific entities
List<ZettelNote> notes = zettelRAG.findByEntities(
    Arrays.asList("Zettelkasten", "Luhmann")
);

// Find notes with specific tags
List<ZettelNote> tagged = zettelRAG.findByTags(
    Arrays.asList("knowledge-management", "productivity")
);
```

#### 7. **ZettelkastenChunking** - Atomic Idea Detection
```java
ChunkingStrategy atomicChunking = new ZettelkastenChunking(
    minChunkSize: 100,
    maxChunkSize: 500,
    targetChunkSize: 300  // Optimal for linking
);

// Splits on:
// - Markdown headings (#{1,6})
// - List boundaries
// - Logical connectors ("therefore", "however", "furthermore")

List<String> atomicIdeas = atomicChunking.chunk(longDocument);
// Each chunk = 1 linkable concept (Zettelkasten principle)
```

### Complete Example

See `ZettelkastenExample.java` for a comprehensive 11-step demonstration:

1. Create 5 atomic notes with automatic link generation
2. Explore auto-generated links (outgoing, incoming, types)
3. View graph statistics (nodes, edges, connectivity)
4. Traverse knowledge graph (BFS from starting note)
5. Find shortest paths between concepts
6. Identify knowledge hubs (most connected notes)
7. Link-aware context retrieval (graph-expanded search)
8. Entity-based search
9. Tag-based search
10. Link type distribution analysis
11. Detect connected components (knowledge clusters)

### Architecture

```
┌─────────────────────────────────────────────────────┐
│          ZettelkastenRAGService (Main API)          │
│  - createNote() with auto-linking                   │
│  - retrieveContextWithLinks()                       │
│  - findByEntities() / findByTags()                  │
└────────────────┬────────────────────────────────────┘
                 │
      ┌──────────┼──────────┐
      │          │          │
┌─────▼────┐ ┌──▼───────┐ ┌▼────────────┐
│ ZettelNote│ │LinkStorage│ │ Zettel-     │
│  (Notes)  │ │ (Persist) │ │ kastenGraph │
│ - Atomic  │ │ - Cache   │ │ - BFS/DFS   │
│ - Links   │ │ - RAG idx │ │ - Paths     │
└───────────┘ └───────────┘ │ - Clusters  │
                             └─────────────┘
      ┌──────────┴──────────┐
      │                     │
┌─────▼────────┐   ┌────────▼──────┐
│EntityExtractor│   │Zettelkasten-  │
│- OpenNLP NER │   │  Chunking     │
│- Patterns    │   │- Atomic ideas │
└──────────────┘   └───────────────┘
```

### Zettelkasten Principles Implemented

1. **Atomicity** - Each note represents one concept
2. **Connectivity** - Bidirectional links between notes
3. **Emergent Structure** - Graph forms organically through linking
4. **Link Types** - Explicit relationship semantics
5. **Discoverability** - Graph algorithms for knowledge exploration

---

## 🦉 OWL Ontology Graph Chunking

The library includes specialized **OWL (Web Ontology Language) graph chunking** capabilities for processing and breaking down large ontologies into manageable, semantically coherent chunks. This feature is designed for knowledge graph applications and is compatible with **Protege plugin architecture** (no Protege dependencies required).

### Key Features

✅ **Multiple Input Methods** - Load from File, InputStream, or OWLOntology object  
✅ **7 Chunking Strategies** - Class-based, namespace, size, depth, module extraction, connected components, annotation-based  
✅ **Protege Compatible** - Designed for integration as Protege plugin (zero Protege dependencies)  
✅ **Semantic Coherence** - Maintains logical relationships within chunks  
✅ **Metadata Tracking** - Each chunk includes rich metadata about its contents  
✅ **Extensible Design** - Easy to add custom OWL chunking strategies  

### Available OWL Chunking Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **CLASS_BASED** | Chunks by class hierarchies preserving inheritance | Ontologies with clear class structures |
| **NAMESPACE_BASED** | Chunks by namespace/URI prefixes | Multi-namespace ontologies |
| **SIZE_BASED** | Chunks by maximum axiom count while maintaining entity coherence | Large ontologies needing size limits |
| **DEPTH_BASED** | Chunks by hierarchy depth levels | Deeply nested class hierarchies |
| **MODULE_EXTRACTION** | Extracts self-contained ontology modules | Modular ontology design |
| **CONNECTED_COMPONENT** | Chunks by graph connectivity analysis | Loosely coupled ontologies |
| **ANNOTATION_BASED** | Chunks by annotation properties (rdfs:label, rdfs:comment) | Well-annotated ontologies |

### Basic Usage

```java
import io.github.vishalmysore.graph.*;
import java.io.File;

// Create chunker with desired strategy
OWLGraphChunker chunker = new OWLGraphChunker(OWLChunkingStrategy.CLASS_BASED);

// Load OWL file
File owlFile = new File("path/to/ontology.owl");
chunker.loadFromFile(owlFile);

// View ontology statistics
System.out.println(chunker.getOntologyStats());

// Perform chunking
List<OWLChunk> chunks = chunker.chunk();

// Process chunks
for (OWLChunk chunk : chunks) {
    System.out.println("Chunk: " + chunk.getId());
    System.out.println("Axioms: " + chunk.getAxiomCount());
    System.out.println("Metadata: " + chunk.getMetadata());
    
    // Get OWL representation
    String owlString = chunk.toOWLString();
}
```

### Load from InputStream (Protege Plugin Use Case)

```java
// Perfect for Protege plugins or when ontology is not a file
try (InputStream inputStream = new FileInputStream(owlFile)) {
    chunker.loadFromStream(inputStream);
    List<OWLChunk> chunks = chunker.chunk();
}
```

### Load Existing OWLOntology Object

```java
// For Protege integration - use the active ontology
OWLOntology ontology = ...; // From Protege API
chunker.loadOntology(ontology);
List<OWLChunk> chunks = chunker.chunk();
```

### Strategy Examples

#### 1. Class-Based Chunking
```java
OWLGraphChunker chunker = new OWLGraphChunker(OWLChunkingStrategy.CLASS_BASED);
chunker.loadFromFile(new File("ontology.owl"));
List<OWLChunk> chunks = chunker.chunk();

// Each chunk contains a top-level class + subclasses + related properties
for (OWLChunk chunk : chunks) {
    String topClass = (String) chunk.getMetadata().get("topClass");
    int relatedClasses = (Integer) chunk.getMetadata().get("relatedClasses");
    System.out.printf("Chunk: %s (Top class: %s, Related: %d)\n", 
        chunk.getId(), topClass, relatedClasses);
}
```

#### 2. Size-Based Chunking
```java
OWLGraphChunker chunker = new OWLGraphChunker(OWLChunkingStrategy.SIZE_BASED);
chunker.setMaxChunkSize(500); // Max 500 axioms per chunk
chunker.loadFromFile(new File("large-ontology.owl"));

List<OWLChunk> chunks = chunker.chunk();
// Each chunk ≤ 500 axioms, entity coherence maintained
```

#### 3. Namespace-Based Chunking
```java
OWLGraphChunker chunker = new OWLGraphChunker(OWLChunkingStrategy.NAMESPACE_BASED);
chunker.loadFromFile(new File("multi-namespace-ontology.owl"));

List<OWLChunk> chunks = chunker.chunk();
// Axioms grouped by their namespace URIs
for (OWLChunk chunk : chunks) {
    String namespace = (String) chunk.getMetadata().get("namespace");
    System.out.println("Namespace: " + namespace);
}
```

#### 4. Module Extraction
```java
OWLGraphChunker chunker = new OWLGraphChunker(OWLChunkingStrategy.MODULE_EXTRACTION);
chunker.loadFromFile(new File("ontology.owl"));

List<OWLChunk> chunks = chunker.chunk();
// Self-contained modules based on syntactic locality
for (OWLChunk chunk : chunks) {
    int seedEntities = (Integer) chunk.getMetadata().get("seedEntityCount");
    System.out.printf("Module with %d seed entities, %d axioms\n", 
        seedEntities, chunk.getAxiomCount());
}
```

### Protege Plugin Integration Example

```java
public class MyProtegePlugin {
    
    public void chunkActiveOntology(OWLOntology ontology) {
        // Use existing ontology from Protege
        OWLGraphChunker chunker = new OWLGraphChunker(
            OWLChunkingStrategy.CLASS_BASED
        );
        
        chunker.loadOntology(ontology);
        List<OWLChunk> chunks = chunker.chunk();
        
        // Process chunks in Protege UI
        displayChunksInUI(chunks);
    }
    
    private void displayChunksInUI(List<OWLChunk> chunks) {
        // Your Protege UI code here
    }
}
```

### OWL Package Structure

```
io.github.vishalmysore.graph/
├── OWLGraphChunker.java          # Main API
├── OWLChunk.java                 # Chunk representation
├── OWLChunkingStrategy.java      # Strategy enum
├── ClassBasedChunker.java        # Class hierarchy chunking
├── NamespaceBasedChunker.java    # Namespace chunking
├── SizeBasedChunker.java         # Size-limited chunking
├── DepthBasedChunker.java        # Depth-based chunking
├── ModuleExtractionChunker.java  # Module extraction
├── ConnectedComponentChunker.java # Graph connectivity
├── AnnotationBasedChunker.java   # Annotation-based
└── OWLGraphChunkerExample.java   # Usage examples
```

### Dependencies

Only **OWL API 5.5.0** is required (automatically added via Maven):

```xml
<dependency>
    <groupId>net.sourceforge.owlapi</groupId>
    <artifactId>owlapi-distribution</artifactId>
    <version>5.5.0</version>
</dependency>
```

### Why OWL Graph Chunking?

- **Large Ontology Processing** - Break down massive ontologies for incremental processing
- **Distributed Processing** - Distribute chunks across multiple workers
- **Focused Analysis** - Analyze specific ontology modules independently
- **RAG for Ontologies** - Index ontology chunks for semantic search over knowledge graphs
- **Protege Extensions** - Build custom Protege plugins with chunking capabilities

---

## Example Output

```
Total documents indexed: 3

=== Semantic Search Results ===
1. [Score: 0.8523] Deep learning uses neural networks with multiple layers
   Author: Bob, Category: AI

2. [Score: 0.7891] Machine learning is a subset of artificial intelligence
   Author: Alice, Category: AI

✓ All data stored locally in: C:\work\navig\lucenerag\lucene-index
✓ No external server required!
```


## Roadmap to Full Agentic Memory

This library currently provides the **foundational RAG infrastructure**. To evolve into a full agentic memory system, the following features are planned for the `io.github.vishalmysore.agenticmemory` package:

### Phase 1: Enhanced Knowledge Organization (Current)
- ✅ Multiple chunking strategies (10+ implementations)
- ✅ Extensible chunking architecture
- ✅ Metadata-rich document storage
- ✅ Hybrid search (vector + keyword)
- ✅ NER-based entity extraction

### Phase 2: Autonomous Note Construction (Planned)
- ⏳ LLM-driven note generation from raw content
- ⏳ Automatic summarization and condensation
- ⏳ Multi-document synthesis
- ⏳ Hierarchical note organization

### Phase 3: Link Generation & Knowledge Graph (Planned)
- ⏳ Automatic relationship detection between documents
- ⏳ Entity co-occurrence analysis
- ⏳ Temporal link tracking
- ⏳ Graph-based traversal API
- ⏳ Semantic link types (supports, contradicts, extends, etc.)

### Phase 4: Dynamic Memory Evolution (Planned)
- ⏳ Adaptive memory consolidation
- ⏳ Importance-based pruning
- ⏳ Temporal decay mechanisms
- ⏳ Knowledge update propagation
- ⏳ Conflict resolution strategies

### Phase 5: Context-Aware Retrieval (Planned)
- ⏳ Query expansion using knowledge graph
- ⏳ Multi-hop reasoning
- ⏳ Contextual relevance scoring
- ⏳ Conversation-aware retrieval
- ⏳ Personalized memory access

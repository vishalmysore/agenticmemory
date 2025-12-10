# Lucene RAG Library

A lightweight, serverless library for Retrieval-Augmented Generation (RAG) operations using Apache Lucene with KnnVectorField for semantic search.

## Features

✅ **No External Server Required** - All data stored locally in Lucene index files  
✅ **Vector Similarity Search** - Using Lucene's KnnFloatVectorField for semantic search  
✅ **Keyword Search** - Traditional full-text search capabilities  
✅ **Hybrid Search** - Combine vector and keyword search with custom weights  
✅ **Metadata Support** - Attach custom metadata to documents  
✅ **Simple API** - Easy to use, clean interface  
✅ **Fully Tested** - Comprehensive unit tests included  

## Storage Architecture

This library uses **local file-based storage** via Apache Lucene:
- **No Qdrant server needed** - Unlike some vector databases, this runs entirely locally
- **No Docker containers** - No external dependencies to manage
- **File-based index** - Lucene creates and manages index files on your local filesystem
- **Embedded solution** - Everything runs in your Java application process

## Dependencies

```xml
<dependencies>
    <dependency>
        <groupId>org.apache.lucene</groupId>
        <artifactId>lucene-core</artifactId>
        <version>9.11.0</version>
    </dependency>
    <dependency>
        <groupId>org.apache.lucene</groupId>
        <artifactId>lucene-analysis-common</artifactId>
        <version>9.11.0</version>
    </dependency>
    <dependency>
        <groupId>org.apache.lucene</groupId>
        <artifactId>lucene-queryparser</artifactId>
        <version>9.11.0</version>
    </dependency>
</dependencies>
```

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

## Running Tests

```bash
mvn test
```

Tests cover:
- Document building and validation
- Vector search accuracy
- Metadata handling
- CRUD operations
- Edge cases and error handling

## Requirements

- Java 18 or higher
- Maven 3.6+
- No external services (Qdrant, Elasticsearch, etc.)

## License

This project is available under standard open source licenses.

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



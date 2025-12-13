package io.github.vishalmysore.rag;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * High-level RAG (Retrieval-Augmented Generation) service that combines
 * document indexing with embedding generation for semantic search.
 */
public class RAGService implements AutoCloseable {

    private final LuceneRAGEngine engine;
    private final EmbeddingProvider embeddingProvider;

    /**
     * Creates a new RAG service.
     *
     * @param indexPath         Path to the Lucene index
     * @param embeddingProvider Provider for generating embeddings
     * @throws IOException if an I/O error occurs
     */
    public RAGService(Path indexPath, EmbeddingProvider embeddingProvider) throws IOException {
        this.embeddingProvider = embeddingProvider;
        this.engine = new LuceneRAGEngine(indexPath, embeddingProvider.getDimension());
    }

    /**
     * Adds a document with automatic embedding generation.
     *
     * @param id       Document ID
     * @param content  Document content
     * @param metadata Optional metadata
     * @throws IOException if an I/O error occurs
     */
    public void addDocument(String id, String content, Map<String, String> metadata) throws IOException {
        float[] embedding = embeddingProvider.embed(content);

        Document.Builder builder = new Document.Builder()
                .id(id)
                .content(content)
                .vector(embedding);

        if (metadata != null) {
            builder.metadata(metadata);
        }

        engine.indexDocument(builder.build());
    }

    /**
     * Adds a document without metadata.
     *
     * @param id      Document ID
     * @param content Document content
     * @throws IOException if an I/O error occurs
     */
    public void addDocument(String id, String content) throws IOException {
        addDocument(id, content, null);
    }

    /**
     * Adds multiple documents in batch.
     *
     * @param documents List of documents to add
     * @throws IOException if an I/O error occurs
     */
    public void addDocuments(List<Document> documents) throws IOException {
        engine.indexDocuments(documents);
    }

    /**
     * Searches for similar documents using semantic search.
     *
     * @param query Query text
     * @param topK  Number of results to return
     * @return List of search results
     * @throws IOException if an I/O error occurs
     */
    public List<SearchResult> search(String query, int topK) throws IOException {
        float[] queryEmbedding = embeddingProvider.embed(query);
        return engine.vectorSearch(queryEmbedding, topK);
    }

    /**
     * Searches using keyword-based search.
     *
     * @param query Query text
     * @param topK  Number of results to return
     * @return List of search results
     * @throws IOException if an I/O error occurs
     */
    public List<SearchResult> keywordSearch(String query, int topK) throws IOException {
        return engine.keywordSearch(query, topK);
    }

    /**
     * Searches using hybrid approach (vector + keyword).
     *
     * @param query        Query text
     * @param topK         Number of results to return
     * @param vectorWeight Weight for vector search (0.0 to 1.0)
     * @return List of search results
     * @throws IOException if an I/O error occurs
     */
    public List<SearchResult> hybridSearch(String query, int topK, float vectorWeight) throws IOException {
        float[] queryEmbedding = embeddingProvider.embed(query);
        return engine.hybridSearch(queryEmbedding, query, topK, vectorWeight);
    }

    /**
     * Retrieves context for RAG by finding the most relevant documents.
     *
     * @param query Query text
     * @param topK  Number of documents to retrieve
     * @return Combined context from top documents
     * @throws IOException if an I/O error occurs
     */
    public String retrieveContext(String query, int topK) throws IOException {
        List<SearchResult> results = search(query, topK);
        StringBuilder context = new StringBuilder();

        for (int i = 0; i < results.size(); i++) {
            if (i > 0) {
                context.append("\n\n---\n\n");
            }
            context.append(results.get(i).getContent());
        }

        return context.toString();
    }

    /**
     * Deletes a document by ID.
     *
     * @param documentId The document ID
     * @throws IOException if an I/O error occurs
     */
    public void deleteDocument(String documentId) throws IOException {
        engine.deleteDocument(documentId);
    }

    /**
     * Retrieves a document by its ID.
     *
     * @param documentId The ID of the document to retrieve
     * @return The document, or null if not found
     * @throws IOException if an I/O error occurs
     */
    public Document getDocumentById(String documentId) throws IOException {
        return engine.getDocumentById(documentId);
    }

    /**
     * Retrieves all documents from the index.
     *
     * @return List of all documents
     * @throws IOException if an I/O error occurs
     */
    public List<Document> getAllDocuments() throws IOException {
        return engine.getAllDocuments();
    }

    /**
     * Checks if a document with the given ID exists.
     *
     * @param documentId The ID to check
     * @return true if the document exists, false otherwise
     * @throws IOException if an I/O error occurs
     */
    public boolean documentExists(String documentId) throws IOException {
        return engine.documentExists(documentId);
    }

    /**
     * Gets the total number of documents in the index.
     *
     * @return The number of documents
     * @throws IOException if an I/O error occurs
     */
    public int getDocumentCount() throws IOException {
        return engine.getDocumentCount();
    }

    /**
     * Commits all pending changes.
     *
     * @throws IOException if an I/O error occurs
     */
    public void commit() throws IOException {
        engine.commit();
    }

    /**
     * Adds a document with automatic chunking using the specified strategy.
     * Each chunk is indexed as a separate document with a generated ID.
     *
     * @param baseId           Base ID for the document (chunks will be named baseId_chunk_1, baseId_chunk_2, etc.)
     * @param content          Document content to be chunked
     * @param chunkingStrategy Strategy to use for chunking
     * @return Number of chunks created
     * @throws IOException if an I/O error occurs
     */
    public int addDocumentWithChunking(String baseId, String content, ChunkingStrategy chunkingStrategy) throws IOException {
        return addDocumentWithChunking(baseId, content, chunkingStrategy, null);
    }

    /**
     * Adds a document with automatic chunking using the specified strategy.
     * Each chunk is indexed as a separate document with metadata.
     *
     * @param baseId           Base ID for the document
     * @param content          Document content to be chunked
     * @param chunkingStrategy Strategy to use for chunking
     * @param metadata         Optional metadata to attach to all chunks
     * @return Number of chunks created
     * @throws IOException if an I/O error occurs
     */
    public int addDocumentWithChunking(String baseId, String content, ChunkingStrategy chunkingStrategy, 
                                      Map<String, String> metadata) throws IOException {
        List<String> chunks = chunkingStrategy.chunk(content);
        
        for (int i = 0; i < chunks.size(); i++) {
            String chunkId = baseId + "_chunk_" + (i + 1);
            addDocument(chunkId, chunks.get(i), metadata);
        }
        
        return chunks.size();
    }

    /**
     * Adds a document from an InputStream with automatic chunking using the specified strategy.
     * The stream is read using UTF-8 encoding.
     *
     * @param baseId           Base ID for the document
     * @param inputStream      InputStream containing document content to be chunked
     * @param chunkingStrategy Strategy to use for chunking
     * @return Number of chunks created
     * @throws IOException if an I/O error occurs
     */
    public int addDocumentWithChunking(String baseId, java.io.InputStream inputStream, 
                                      ChunkingStrategy chunkingStrategy) throws IOException {
        return addDocumentWithChunking(baseId, inputStream, chunkingStrategy, null);
    }

    /**
     * Adds a document from an InputStream with automatic chunking using the specified strategy.
     * Each chunk is indexed as a separate document with metadata.
     * The stream is read using UTF-8 encoding.
     *
     * @param baseId           Base ID for the document
     * @param inputStream      InputStream containing document content to be chunked
     * @param chunkingStrategy Strategy to use for chunking
     * @param metadata         Optional metadata to attach to all chunks
     * @return Number of chunks created
     * @throws IOException if an I/O error occurs
     */
    public int addDocumentWithChunking(String baseId, java.io.InputStream inputStream, 
                                      ChunkingStrategy chunkingStrategy,
                                      Map<String, String> metadata) throws IOException {
        List<String> chunks = chunkingStrategy.chunk(inputStream);
        
        for (int i = 0; i < chunks.size(); i++) {
            String chunkId = baseId + "_chunk_" + (i + 1);
            addDocument(chunkId, chunks.get(i), metadata);
        }
        
        return chunks.size();
    }

    /**
     * Adds a document from an InputStream with automatic chunking using the specified strategy.
     * Each chunk is indexed as a separate document with metadata.
     *
     * @param baseId           Base ID for the document
     * @param inputStream      InputStream containing document content to be chunked
     * @param chunkingStrategy Strategy to use for chunking
     * @param charsetName      Character encoding to use (e.g., "UTF-8", "ISO-8859-1")
     * @param metadata         Optional metadata to attach to all chunks
     * @return Number of chunks created
     * @throws IOException if an I/O error occurs
     */
    public int addDocumentWithChunking(String baseId, java.io.InputStream inputStream, 
                                      ChunkingStrategy chunkingStrategy,
                                      String charsetName,
                                      Map<String, String> metadata) throws IOException {
        List<String> chunks = chunkingStrategy.chunk(inputStream, charsetName);
        
        for (int i = 0; i < chunks.size(); i++) {
            String chunkId = baseId + "_chunk_" + (i + 1);
            addDocument(chunkId, chunks.get(i), metadata);
        }
        
        return chunks.size();
    }

    @Override
    public void close() throws IOException {
        engine.close();
    }
}


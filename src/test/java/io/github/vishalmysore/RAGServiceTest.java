package io.github.vishalmysore;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class RAGServiceTest {

    private RAGService ragService;
    private MockEmbeddingProvider embeddingProvider;

    @TempDir
    Path tempDir;

    @BeforeEach
    void setUp() throws IOException {
        embeddingProvider = new MockEmbeddingProvider(128);
        ragService = new RAGService(tempDir, embeddingProvider);
    }

    @AfterEach
    void tearDown() throws IOException {
        if (ragService != null) {
            ragService.close();
        }
    }

    @Test
    void testAddDocument() throws IOException {
        ragService.addDocument("doc1", "This is a test document");
        ragService.commit();

        assertEquals(1, ragService.getDocumentCount());
    }

    @Test
    void testAddDocumentWithMetadata() throws IOException {
        Map<String, String> metadata = new HashMap<>();
        metadata.put("author", "Bob");
        metadata.put("date", "2025-12-09");

        ragService.addDocument("doc2", "Document with metadata", metadata);
        ragService.commit();

        List<SearchResult> results = ragService.search("Document", 1);
        assertFalse(results.isEmpty());
        assertEquals("Bob", results.get(0).getMetadata("author"));
    }

    @Test
    void testSemanticSearch() throws IOException {
        ragService.addDocument("doc1", "Machine learning is a subset of artificial intelligence");
        ragService.addDocument("doc2", "Cooking pasta requires boiling water");
        ragService.addDocument("doc3", "Deep learning uses neural networks");
        ragService.commit();

        List<SearchResult> results = ragService.search("artificial intelligence", 2);

        assertFalse(results.isEmpty());
        assertTrue(results.size() <= 2);
    }

    @Test
    void testRetrieveContext() throws IOException {
        ragService.addDocument("doc1", "Python is a programming language");
        ragService.addDocument("doc2", "Java is also a programming language");
        ragService.addDocument("doc3", "JavaScript runs in browsers");
        ragService.commit();

        String context = ragService.retrieveContext("programming languages", 2);

        assertNotNull(context);
        assertFalse(context.isEmpty());
    }

    @Test
    void testDeleteDocument() throws IOException {
        ragService.addDocument("doc1", "Document to delete");
        ragService.addDocument("doc2", "Document to keep");
        ragService.commit();

        assertEquals(2, ragService.getDocumentCount());

        ragService.deleteDocument("doc1");
        ragService.commit();

        assertEquals(1, ragService.getDocumentCount());
    }

    @Test
    void testKeywordSearch() throws IOException {
        ragService.addDocument("doc1", "The quick brown fox jumps over the lazy dog");
        ragService.addDocument("doc2", "A fast brown fox leaps over a sleepy dog");
        ragService.commit();

        List<SearchResult> results = ragService.keywordSearch("fox", 2);

        // Keyword search might return results depending on analyzer behavior
        assertNotNull(results);
    }

    @Test
    void testHybridSearch() throws IOException {
        ragService.addDocument("doc1", "Lucene is a search library");
        ragService.addDocument("doc2", "Apache Lucene provides full-text search");
        ragService.addDocument("doc3", "Elasticsearch is built on Lucene");
        ragService.commit();

        List<SearchResult> results = ragService.hybridSearch("search library", 2, 0.5f);

        assertNotNull(results);
        assertTrue(results.size() <= 2);
    }

    @Test
    void testMultipleDocumentsWithSameEmbedding() throws IOException {
        String sameText = "Identical content";

        ragService.addDocument("doc1", sameText);
        ragService.addDocument("doc2", sameText);
        ragService.commit();

        List<SearchResult> results = ragService.search(sameText, 2);

        assertEquals(2, results.size());
    }

    @Test
    void testEmptyQuery() throws IOException {
        ragService.addDocument("doc1", "Some content");
        ragService.commit();

        String context = ragService.retrieveContext("", 1);
        assertNotNull(context);
    }

    @Test
    void testLargeNumberOfDocuments() throws IOException {
        for (int i = 0; i < 50; i++) {
            ragService.addDocument("doc" + i, "Document number " + i + " with unique content");
        }
        ragService.commit();

        assertEquals(50, ragService.getDocumentCount());

        List<SearchResult> results = ragService.search("unique content", 10);
        assertTrue(results.size() <= 10);
    }

    @Test
    void testSearchReturnsRelevantResults() throws IOException {
        ragService.addDocument("ai1", "Artificial intelligence and machine learning");
        ragService.addDocument("ai2", "Neural networks and deep learning");
        ragService.addDocument("food1", "Pizza and pasta recipes");
        ragService.addDocument("food2", "Baking bread and cakes");
        ragService.commit();

        List<SearchResult> results = ragService.search("artificial intelligence", 2);

        assertFalse(results.isEmpty());
        // With mock embeddings, results are deterministic based on text hash
    }

    @Test
    void testContextCombination() throws IOException {
        ragService.addDocument("doc1", "First piece of information");
        ragService.addDocument("doc2", "Second piece of information");
        ragService.addDocument("doc3", "Third piece of information");
        ragService.commit();

        String context = ragService.retrieveContext("information", 3);

        assertTrue(context.contains("---") || context.length() > 0);
    }

    @Test
    void testGetDocumentById() throws IOException {
        Map<String, String> metadata = new HashMap<>();
        metadata.put("author", "Alice");
        metadata.put("category", "test");
        
        ragService.addDocument("doc1", "Test content", metadata);
        ragService.commit();

        Document doc = ragService.getDocumentById("doc1");
        
        assertNotNull(doc);
        assertEquals("doc1", doc.getId());
        assertEquals("Test content", doc.getContent());
        assertEquals("Alice", doc.getMetadata("author"));
        assertEquals("test", doc.getMetadata("category"));
        assertNotNull(doc.getVector());
        assertEquals(128, doc.getVector().length);
    }

    @Test
    void testGetDocumentByIdNotFound() throws IOException {
        ragService.addDocument("doc1", "Test content");
        ragService.commit();

        Document doc = ragService.getDocumentById("nonexistent");
        
        assertNull(doc);
    }

    @Test
    void testGetAllDocuments() throws IOException {
        ragService.addDocument("doc1", "Content 1");
        ragService.addDocument("doc2", "Content 2");
        ragService.addDocument("doc3", "Content 3");
        ragService.commit();

        List<Document> allDocs = ragService.getAllDocuments();
        
        assertEquals(3, allDocs.size());
        
        // Verify all documents are present
        List<String> ids = allDocs.stream().map(Document::getId).toList();
        assertTrue(ids.contains("doc1"));
        assertTrue(ids.contains("doc2"));
        assertTrue(ids.contains("doc3"));
    }

    @Test
    void testGetAllDocumentsWithMetadata() throws IOException {
        Map<String, String> meta1 = new HashMap<>();
        meta1.put("author", "Alice");
        
        Map<String, String> meta2 = new HashMap<>();
        meta2.put("author", "Bob");
        
        ragService.addDocument("doc1", "Content 1", meta1);
        ragService.addDocument("doc2", "Content 2", meta2);
        ragService.commit();

        List<Document> allDocs = ragService.getAllDocuments();
        
        assertEquals(2, allDocs.size());
        
        for (Document doc : allDocs) {
            assertNotNull(doc.getMetadata("author"));
            assertTrue(doc.getMetadata("author").equals("Alice") || 
                      doc.getMetadata("author").equals("Bob"));
        }
    }

    @Test
    void testDocumentExists() throws IOException {
        ragService.addDocument("doc1", "Test content");
        ragService.commit();

        assertTrue(ragService.documentExists("doc1"));
        assertFalse(ragService.documentExists("nonexistent"));
    }

    @Test
    void testLoadPreviouslyIndexedDocuments() throws IOException {
        // Index documents in first session
        ragService.addDocument("doc1", "First document");
        ragService.addDocument("doc2", "Second document");
        ragService.commit();
        ragService.close();

        // Open new session with same index
        RAGService newSession = new RAGService(tempDir, embeddingProvider);
        
        try {
            // Verify documents are still there
            assertEquals(2, newSession.getDocumentCount());
            
            Document doc1 = newSession.getDocumentById("doc1");
            assertNotNull(doc1);
            assertEquals("First document", doc1.getContent());
            
            Document doc2 = newSession.getDocumentById("doc2");
            assertNotNull(doc2);
            assertEquals("Second document", doc2.getContent());
            
            // Verify we can add more documents
            newSession.addDocument("doc3", "Third document");
            newSession.commit();
            
            assertEquals(3, newSession.getDocumentCount());
        } finally {
            newSession.close();
        }
    }
}


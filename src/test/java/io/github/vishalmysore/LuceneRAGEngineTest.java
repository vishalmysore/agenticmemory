package io.github.vishalmysore;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class LuceneRAGEngineTest {

    private LuceneRAGEngine engine;
    private Directory directory;
    private static final int VECTOR_DIM = 128;

    @BeforeEach
    void setUp() throws IOException {
        directory = new ByteBuffersDirectory();
        engine = new LuceneRAGEngine(directory, new org.apache.lucene.analysis.standard.StandardAnalyzer(), VECTOR_DIM);
    }

    @AfterEach
    void tearDown() throws IOException {
        if (engine != null) {
            engine.close();
        }
    }

    @Test
    void testIndexSingleDocument() throws IOException {
        float[] vector = generateRandomVector(VECTOR_DIM);
        Document doc = new Document.Builder()
                .id("test1")
                .content("This is a test document")
                .vector(vector)
                .build();

        engine.indexDocument(doc);
        engine.commit();

        assertEquals(1, engine.getDocumentCount());
    }

    @Test
    void testIndexMultipleDocuments() throws IOException {
        List<Document> docs = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            docs.add(new Document.Builder()
                    .id("doc" + i)
                    .content("Document content " + i)
                    .vector(generateRandomVector(VECTOR_DIM))
                    .build());
        }

        engine.indexDocuments(docs);
        assertEquals(5, engine.getDocumentCount());
    }

    @Test
    void testVectorSearch() throws IOException {
        // Index documents
        float[] vector1 = generateRandomVector(VECTOR_DIM);
        float[] vector2 = generateRandomVector(VECTOR_DIM);

        Document doc1 = new Document.Builder()
                .id("doc1")
                .content("First document")
                .vector(vector1)
                .build();

        Document doc2 = new Document.Builder()
                .id("doc2")
                .content("Second document")
                .vector(vector2)
                .build();

        engine.indexDocument(doc1);
        engine.indexDocument(doc2);
        engine.commit();

        // Search with vector1 - should return doc1 first
        List<SearchResult> results = engine.vectorSearch(vector1, 2);
        assertFalse(results.isEmpty());
        assertEquals("doc1", results.get(0).getId());
    }

    @Test
    void testDocumentWithMetadata() throws IOException {
        Map<String, String> metadata = new HashMap<>();
        metadata.put("author", "Alice");
        metadata.put("category", "science");

        Document doc = new Document.Builder()
                .id("meta_doc")
                .content("Document with metadata")
                .vector(generateRandomVector(VECTOR_DIM))
                .metadata(metadata)
                .build();

        engine.indexDocument(doc);
        engine.commit();

        List<SearchResult> results = engine.vectorSearch(generateRandomVector(VECTOR_DIM), 1);
        assertFalse(results.isEmpty());
        assertEquals("Alice", results.get(0).getMetadata("author"));
        assertEquals("science", results.get(0).getMetadata("category"));
    }

    @Test
    void testUpdateDocument() throws IOException {
        // Index original document
        Document original = new Document.Builder()
                .id("update_test")
                .content("Original content")
                .vector(generateRandomVector(VECTOR_DIM))
                .build();

        engine.indexDocument(original);
        engine.commit();
        assertEquals(1, engine.getDocumentCount());

        // Update the document
        Document updated = new Document.Builder()
                .id("update_test")
                .content("Updated content")
                .vector(generateRandomVector(VECTOR_DIM))
                .build();

        engine.indexDocument(updated);
        engine.commit();

        // Should still have only 1 document
        assertEquals(1, engine.getDocumentCount());

        List<SearchResult> results = engine.vectorSearch(generateRandomVector(VECTOR_DIM), 1);
        assertEquals("Updated content", results.get(0).getContent());
    }

    @Test
    void testDeleteDocument() throws IOException {
        Document doc = new Document.Builder()
                .id("delete_test")
                .content("To be deleted")
                .vector(generateRandomVector(VECTOR_DIM))
                .build();

        engine.indexDocument(doc);
        engine.commit();
        assertEquals(1, engine.getDocumentCount());

        engine.deleteDocument("delete_test");
        engine.commit();
        assertEquals(0, engine.getDocumentCount());
    }

    @Test
    void testVectorDimensionMismatch() {
        float[] wrongVector = new float[64]; // Wrong dimension

        Document doc = new Document.Builder()
                .id("wrong_dim")
                .content("Wrong dimension vector")
                .vector(wrongVector)
                .build();

        assertThrows(IllegalArgumentException.class, () -> {
            engine.indexDocument(doc);
        });
    }

    @Test
    void testQueryVectorDimensionMismatch() {
        float[] wrongQueryVector = new float[64]; // Wrong dimension

        assertThrows(IllegalArgumentException.class, () -> {
            engine.vectorSearch(wrongQueryVector, 10);
        });
    }

    @Test
    void testEmptyIndex() throws IOException {
        assertEquals(0, engine.getDocumentCount());

        List<SearchResult> results = engine.vectorSearch(generateRandomVector(VECTOR_DIM), 10);
        assertTrue(results.isEmpty());
    }

    @Test
    void testWithFileSystemDirectory(@TempDir Path tempDir) throws IOException {
        LuceneRAGEngine fsEngine = new LuceneRAGEngine(tempDir, VECTOR_DIM);

        Document doc = new Document.Builder()
                .id("fs_test")
                .content("File system test")
                .vector(generateRandomVector(VECTOR_DIM))
                .build();

        fsEngine.indexDocument(doc);
        fsEngine.commit();

        assertEquals(1, fsEngine.getDocumentCount());
        fsEngine.close();
    }

    @Test
    void testHybridSearch() throws IOException {
        float[] vector1 = generateRandomVector(VECTOR_DIM);

        Document doc1 = new Document.Builder()
                .id("hybrid1")
                .content("machine learning artificial intelligence")
                .vector(vector1)
                .build();

        Document doc2 = new Document.Builder()
                .id("hybrid2")
                .content("cooking recipes food")
                .vector(generateRandomVector(VECTOR_DIM))
                .build();

        engine.indexDocument(doc1);
        engine.indexDocument(doc2);
        engine.commit();

        List<SearchResult> results = engine.hybridSearch(
                vector1,
                "machine",
                2,
                0.7f
        );

        assertFalse(results.isEmpty());
    }

    private float[] generateRandomVector(int dimension) {
        Random random = new Random();
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = random.nextFloat();
        }
        // Normalize
        float magnitude = 0;
        for (float v : vector) {
            magnitude += v * v;
        }
        magnitude = (float) Math.sqrt(magnitude);
        for (int i = 0; i < dimension; i++) {
            vector[i] /= magnitude;
        }
        return vector;
    }
}


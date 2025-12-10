package io.github.vishalmysore;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.HashMap;
import java.util.Map;

class DocumentTest {

    @Test
    void testDocumentBuilder() {
        float[] vector = {0.1f, 0.2f, 0.3f};
        Map<String, String> metadata = new HashMap<>();
        metadata.put("author", "John Doe");
        metadata.put("category", "technology");

        Document doc = new Document.Builder()
                .id("doc1")
                .content("This is a test document")
                .vector(vector)
                .metadata(metadata)
                .build();

        assertEquals("doc1", doc.getId());
        assertEquals("This is a test document", doc.getContent());
        assertArrayEquals(vector, doc.getVector());
        assertEquals("John Doe", doc.getMetadata("author"));
        assertEquals("technology", doc.getMetadata("category"));
    }

    @Test
    void testDocumentBuilderWithAddMetadata() {
        Document doc = new Document.Builder()
                .id("doc2")
                .content("Another document")
                .addMetadata("key1", "value1")
                .addMetadata("key2", "value2")
                .build();

        assertEquals("value1", doc.getMetadata("key1"));
        assertEquals("value2", doc.getMetadata("key2"));
    }

    @Test
    void testDocumentWithoutVector() {
        Document doc = new Document.Builder()
                .id("doc3")
                .content("Document without vector")
                .build();

        assertEquals("doc3", doc.getId());
        assertEquals("Document without vector", doc.getContent());
        assertNull(doc.getVector());
    }

    @Test
    void testDocumentBuilderRequiresId() {
        assertThrows(IllegalArgumentException.class, () -> {
            new Document.Builder()
                    .content("No ID document")
                    .build();
        });
    }

    @Test
    void testDocumentBuilderRequiresContent() {
        assertThrows(IllegalArgumentException.class, () -> {
            new Document.Builder()
                    .id("doc4")
                    .build();
        });
    }

    @Test
    void testEmptyIdThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> {
            new Document.Builder()
                    .id("")
                    .content("Content")
                    .build();
        });
    }
}


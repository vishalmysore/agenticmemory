package io.github.vishalmysore;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.HashMap;
import java.util.Map;

class SearchResultTest {

    @Test
    void testSearchResultCreation() {
        Map<String, String> metadata = new HashMap<>();
        metadata.put("author", "Jane Doe");

        SearchResult result = new SearchResult(
                "result1",
                "Sample content",
                0.95f,
                metadata
        );

        assertEquals("result1", result.getId());
        assertEquals("Sample content", result.getContent());
        assertEquals(0.95f, result.getScore());
        assertEquals("Jane Doe", result.getMetadata("author"));
    }

    @Test
    void testSearchResultWithoutMetadata() {
        SearchResult result = new SearchResult(
                "result2",
                "Another content",
                0.85f,
                null
        );

        assertEquals("result2", result.getId());
        assertNull(result.getMetadata("any_key"));
    }

    @Test
    void testSearchResultToString() {
        Map<String, String> metadata = new HashMap<>();
        metadata.put("source", "test");

        SearchResult result = new SearchResult(
                "result3",
                "Test content",
                0.75f,
                metadata
        );

        String str = result.toString();
        assertTrue(str.contains("result3"));
        assertTrue(str.contains("0.75"));
        assertTrue(str.contains("Test content"));
    }
}


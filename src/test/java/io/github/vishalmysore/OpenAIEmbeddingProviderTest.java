package io.github.vishalmysore;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import static org.junit.jupiter.api.Assertions.*;

class OpenAIEmbeddingProviderTest {

    @Test
    void testConstructorValidation() {
        assertThrows(IllegalArgumentException.class, () -> 
            new OpenAIEmbeddingProvider(null, "key", "model", 1536)
        );
        
        assertThrows(IllegalArgumentException.class, () -> 
            new OpenAIEmbeddingProvider("", "key", "model", 1536)
        );
        
        assertThrows(IllegalArgumentException.class, () -> 
            new OpenAIEmbeddingProvider("url", null, "model", 1536)
        );
        
        assertThrows(IllegalArgumentException.class, () -> 
            new OpenAIEmbeddingProvider("url", "", "model", 1536)
        );
        
        assertThrows(IllegalArgumentException.class, () -> 
            new OpenAIEmbeddingProvider("url", "key", null, 1536)
        );
        
        assertThrows(IllegalArgumentException.class, () -> 
            new OpenAIEmbeddingProvider("url", "key", "", 1536)
        );
        
        assertThrows(IllegalArgumentException.class, () -> 
            new OpenAIEmbeddingProvider("url", "key", "model", 0)
        );
        
        assertThrows(IllegalArgumentException.class, () -> 
            new OpenAIEmbeddingProvider("url", "key", "model", -1)
        );
    }

    @Test
    void testValidConstruction() {
        OpenAIEmbeddingProvider provider = new OpenAIEmbeddingProvider(
            "https://api.openai.com/v1/embeddings",
            "test-key",
            "text-embedding-3-small",
            1536
        );
        
        assertNotNull(provider);
        assertEquals(1536, provider.getDimension());
        assertEquals("https://api.openai.com/v1/embeddings", provider.getApiUrl());
        assertEquals("text-embedding-3-small", provider.getModel());
        
        provider.close();
    }

    @Test
    void testDefaultUrlConstructor() {
        OpenAIEmbeddingProvider provider = new OpenAIEmbeddingProvider(
            "test-key",
            "text-embedding-ada-002",
            1536
        );
        
        assertNotNull(provider);
        assertEquals(1536, provider.getDimension());
        assertEquals("https://api.openai.com/v1/embeddings", provider.getApiUrl());
        assertEquals("text-embedding-ada-002", provider.getModel());
        
        provider.close();
    }

    @Test
    void testEmbedWithNullText() {
        OpenAIEmbeddingProvider provider = new OpenAIEmbeddingProvider(
            "test-key",
            "text-embedding-3-small",
            1536
        );
        
        assertThrows(IllegalArgumentException.class, () -> provider.embed(null));
        assertThrows(IllegalArgumentException.class, () -> provider.embed(""));
        assertThrows(IllegalArgumentException.class, () -> provider.embed("   "));
        
        provider.close();
    }

    /**
     * This test only runs if OPENAI_API_KEY environment variable is set.
     * It makes an actual API call to OpenAI.
     */
    @Test
    @EnabledIfEnvironmentVariable(named = "OPENAI_API_KEY", matches = ".+")
    void testActualEmbeddingGeneration() {
        String apiKey = System.getenv("OPENAI_API_KEY");
        
        OpenAIEmbeddingProvider provider = new OpenAIEmbeddingProvider(
            apiKey,
            "text-embedding-3-small",
            1536
        );
        
        String text = "This is a test sentence for embedding generation.";
        float[] embedding = provider.embed(text);
        
        assertNotNull(embedding);
        assertEquals(1536, embedding.length);
        
        // Check that the embedding contains non-zero values
        boolean hasNonZero = false;
        for (float value : embedding) {
            if (value != 0.0f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue(hasNonZero, "Embedding should contain non-zero values");
        
        provider.close();
    }

    /**
     * This test only runs if OPENAI_API_KEY environment variable is set.
     * Tests embedding consistency - same text should produce same embedding.
     */
    @Test
    @EnabledIfEnvironmentVariable(named = "OPENAI_API_KEY", matches = ".+")
    void testEmbeddingConsistency() {
        String apiKey = System.getenv("OPENAI_API_KEY");
        
        OpenAIEmbeddingProvider provider = new OpenAIEmbeddingProvider(
            apiKey,
            "text-embedding-3-small",
            1536
        );
        
        String text = "Consistency test sentence.";
        float[] embedding1 = provider.embed(text);
        float[] embedding2 = provider.embed(text);
        
        assertArrayEquals(embedding1, embedding2, 0.0001f);
        
        provider.close();
    }

    /**
     * This test only runs if OPENAI_API_KEY environment variable is set.
     * Tests with text-embedding-ada-002 model.
     */
    @Test
    @EnabledIfEnvironmentVariable(named = "OPENAI_API_KEY", matches = ".+")
    void testWithAda002Model() {
        String apiKey = System.getenv("OPENAI_API_KEY");
        
        OpenAIEmbeddingProvider provider = new OpenAIEmbeddingProvider(
            apiKey,
            "text-embedding-ada-002",
            1536
        );
        
        String text = "Testing with ada-002 model.";
        float[] embedding = provider.embed(text);
        
        assertNotNull(embedding);
        assertEquals(1536, embedding.length);
        
        provider.close();
    }
}

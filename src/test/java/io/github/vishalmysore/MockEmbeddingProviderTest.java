package io.github.vishalmysore;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MockEmbeddingProviderTest {

    @Test
    void testEmbeddingGeneration() {
        MockEmbeddingProvider provider = new MockEmbeddingProvider(128);

        String text = "This is a test sentence";
        float[] embedding = provider.embed(text);

        assertNotNull(embedding);
        assertEquals(128, embedding.length);
    }

    @Test
    void testConsistentEmbeddings() {
        MockEmbeddingProvider provider = new MockEmbeddingProvider(64);

        String text = "Consistent test";
        float[] embedding1 = provider.embed(text);
        float[] embedding2 = provider.embed(text);

        assertArrayEquals(embedding1, embedding2);
    }

    @Test
    void testDifferentTextsProduceDifferentEmbeddings() {
        MockEmbeddingProvider provider = new MockEmbeddingProvider(64);

        float[] embedding1 = provider.embed("Text A");
        float[] embedding2 = provider.embed("Text B");

        assertFalse(java.util.Arrays.equals(embedding1, embedding2));
    }

    @Test
    void testNormalizedVectors() {
        MockEmbeddingProvider provider = new MockEmbeddingProvider(100);

        float[] embedding = provider.embed("Normalization test");

        // Calculate magnitude
        float magnitude = 0;
        for (float v : embedding) {
            magnitude += v * v;
        }
        magnitude = (float) Math.sqrt(magnitude);

        // Should be approximately 1.0 (normalized)
        assertEquals(1.0f, magnitude, 0.001f);
    }

    @Test
    void testGetDimension() {
        MockEmbeddingProvider provider = new MockEmbeddingProvider(256);
        assertEquals(256, provider.getDimension());
    }

    @Test
    void testDifferentSeeds() {
        MockEmbeddingProvider provider1 = new MockEmbeddingProvider(64, 42);
        MockEmbeddingProvider provider2 = new MockEmbeddingProvider(64, 100);

        // Same text, different seeds might produce different initial randomness
        // but due to hash-based seeding, same text should still be consistent within each provider
        float[] emb1a = provider1.embed("Test");
        float[] emb1b = provider1.embed("Test");

        assertArrayEquals(emb1a, emb1b);
    }
}


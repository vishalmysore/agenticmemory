package io.github.vishalmysore;

import java.util.Random;

/**
 * Simple mock embedding provider for testing and demonstration purposes.
 * In production, replace this with a real embedding model (OpenAI, Sentence Transformers, etc.)
 */
public class MockEmbeddingProvider implements EmbeddingProvider {

    private final int dimension;
    private final Random random;

    public MockEmbeddingProvider(int dimension) {
        this(dimension, 42); // Default seed for reproducibility
    }

    public MockEmbeddingProvider(int dimension, long seed) {
        this.dimension = dimension;
        this.random = new Random(seed);
    }

    @Override
    public float[] embed(String text) {
        // Simple hash-based embedding for consistent results
        random.setSeed(text.hashCode());
        float[] embedding = new float[dimension];

        for (int i = 0; i < dimension; i++) {
            embedding[i] = random.nextFloat() * 2 - 1; // Range [-1, 1]
        }

        // Normalize the vector
        normalize(embedding);
        return embedding;
    }

    @Override
    public int getDimension() {
        return dimension;
    }

    private void normalize(float[] vector) {
        float magnitude = 0;
        for (float v : vector) {
            magnitude += v * v;
        }
        magnitude = (float) Math.sqrt(magnitude);

        if (magnitude > 0) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= magnitude;
            }
        }
    }
}


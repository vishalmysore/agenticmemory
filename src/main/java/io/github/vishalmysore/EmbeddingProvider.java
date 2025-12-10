package io.github.vishalmysore;

/**
 * Interface for generating text embeddings.
 * Implementations can use various embedding models (OpenAI, HuggingFace, etc.)
 */
public interface EmbeddingProvider {

    /**
     * Generates a vector embedding for the given text.
     *
     * @param text The text to embed
     * @return The vector embedding
     */
    float[] embed(String text);

    /**
     * Gets the dimension of the embeddings produced by this provider.
     *
     * @return The embedding dimension
     */
    int getDimension();
}


package io.github.vishalmysore.rag;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Interface for implementing custom chunking strategies in RAG systems.
 * 
 * Chunking strategies determine how to split large documents into smaller,
 * semantically meaningful units for indexing and retrieval.
 * 
 * Implementations should be stateless and thread-safe.
 * 
 * Example usage:
 * <pre>
 * ChunkingStrategy strategy = new SlidingWindowChunking(150, 30);
 * List&lt;String&gt; chunks = strategy.chunk(documentContent);
 * 
 * // Or from InputStream
 * try (InputStream is = new FileInputStream("document.txt")) {
 *     List&lt;String&gt; chunks = strategy.chunk(is);
 * }
 * </pre>
 */
public interface ChunkingStrategy {
    
    /**
     * Splits the input content into chunks according to this strategy's algorithm.
     * 
     * @param content The text content to be chunked
     * @return A list of text chunks. Each chunk should be a standalone, meaningful unit.
     *         Empty list if content is null or empty.
     */
    List<String> chunk(String content);
    
    /**
     * Splits the input stream content into chunks according to this strategy's algorithm.
     * 
     * Default implementation reads the entire stream into a String and delegates to chunk(String).
     * Implementations may override this for streaming processing of large files.
     * 
     * @param inputStream The input stream containing text content to be chunked (UTF-8 encoding assumed)
     * @return A list of text chunks. Each chunk should be a standalone, meaningful unit.
     *         Empty list if stream is empty or null.
     * @throws IOException If an I/O error occurs while reading the stream
     */
    default List<String> chunk(InputStream inputStream) throws IOException {
        if (inputStream == null) {
            return chunk((String) null);
        }
        
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
            String content = reader.lines().collect(Collectors.joining("\n"));
            return chunk(content);
        }
    }
    
    /**
     * Splits the input stream content into chunks with custom character encoding.
     * 
     * @param inputStream The input stream containing text content to be chunked
     * @param charsetName The name of the charset to use for decoding (e.g., "UTF-8", "ISO-8859-1")
     * @return A list of text chunks. Each chunk should be a standalone, meaningful unit.
     *         Empty list if stream is empty or null.
     * @throws IOException If an I/O error occurs while reading the stream
     */
    default List<String> chunk(InputStream inputStream, String charsetName) throws IOException {
        if (inputStream == null) {
            return chunk((String) null);
        }
        
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(inputStream, charsetName))) {
            String content = reader.lines().collect(Collectors.joining("\n"));
            return chunk(content);
        }
    }
    
    /**
     * Returns a human-readable name for this chunking strategy.
     * 
     * @return The strategy name (e.g., "Sliding Window", "Entity-Based")
     */
    String getName();
    
    /**
     * Returns a detailed description of how this strategy works.
     * 
     * @return A description explaining the chunking algorithm and its use cases
     */
    String getDescription();
}

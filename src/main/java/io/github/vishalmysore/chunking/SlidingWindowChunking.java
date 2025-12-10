package io.github.vishalmysore.chunking;

import io.github.vishalmysore.ChunkingStrategy;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Sliding Window Chunking Strategy
 * 
 * Creates overlapping chunks by sliding a fixed-size window across the text.
 * Overlap ensures context is preserved across chunk boundaries.
 * 
 * Best for: Healthcare records, continuous narratives, patient notes
 * 
 * Parameters:
 * - windowSize: Number of words per chunk
 * - overlap: Number of words to overlap between consecutive chunks
 */
public class SlidingWindowChunking implements ChunkingStrategy {
    
    private final int windowSize;
    private final int overlap;
    
    /**
     * Creates a sliding window chunking strategy.
     * 
     * @param windowSize Number of words per chunk (e.g., 150)
     * @param overlap Number of overlapping words (e.g., 30 for 20% overlap)
     */
    public SlidingWindowChunking(int windowSize, int overlap) {
        if (windowSize <= 0) {
            throw new IllegalArgumentException("Window size must be positive");
        }
        if (overlap < 0 || overlap >= windowSize) {
            throw new IllegalArgumentException("Overlap must be between 0 and window size");
        }
        this.windowSize = windowSize;
        this.overlap = overlap;
    }
    
    @Override
    public List<String> chunk(String content) {
        if (content == null || content.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        String[] words = content.split("\\s+");
        List<String> chunks = new ArrayList<>();
        int step = windowSize - overlap;
        
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + windowSize, words.length);
            String chunk = String.join(" ", Arrays.copyOfRange(words, i, end));
            chunks.add(chunk);
            
            if (end >= words.length) break;
        }
        
        return chunks;
    }
    
    @Override
    public String getName() {
        return "Sliding Window Chunking";
    }
    
    @Override
    public String getDescription() {
        return String.format("Creates overlapping chunks with window size of %d words and %d words overlap (%.1f%%)",
                           windowSize, overlap, (overlap * 100.0 / windowSize));
    }
    
    public int getWindowSize() {
        return windowSize;
    }
    
    public int getOverlap() {
        return overlap;
    }
}

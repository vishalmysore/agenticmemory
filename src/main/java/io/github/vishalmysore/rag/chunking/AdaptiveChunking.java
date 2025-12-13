package io.github.vishalmysore.rag.chunking;

import io.github.vishalmysore.rag.ChunkingStrategy;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Adaptive Chunking Strategy
 * 
 * Dynamically adjusts chunk sizes based on document structure and boundaries (e.g., sections, paragraphs).
 * Uses regex patterns to detect section boundaries and creates chunks accordingly.
 */
public class AdaptiveChunking implements ChunkingStrategy {
    private final Pattern sectionPattern;
    private final int minChunkSize;
    private final int maxChunkSize;

    /**
     * Creates an AdaptiveChunking strategy with default size constraints.
     * 
     * @param sectionRegex Regular expression pattern to identify section boundaries
     */
    public AdaptiveChunking(String sectionRegex) {
        this(sectionRegex, 200, 1000);
    }

    /**
     * Creates an AdaptiveChunking strategy with custom size constraints.
     * 
     * @param sectionRegex Regular expression pattern to identify section boundaries
     * @param minChunkSize Minimum chunk size in characters
     * @param maxChunkSize Maximum chunk size in characters
     */
    public AdaptiveChunking(String sectionRegex, int minChunkSize, int maxChunkSize) {
        this.sectionPattern = Pattern.compile(sectionRegex);
        this.minChunkSize = minChunkSize;
        this.maxChunkSize = maxChunkSize;
    }

    @Override
    public List<String> chunk(String content) {
        List<String> chunks = new ArrayList<>();
        
        // Find all section boundaries
        Matcher matcher = sectionPattern.matcher(content);
        List<Integer> boundaries = new ArrayList<>();
        boundaries.add(0); // Start of document
        
        while (matcher.find()) {
            boundaries.add(matcher.start());
        }
        boundaries.add(content.length()); // End of document
        
        // Create chunks based on boundaries
        for (int i = 0; i < boundaries.size() - 1; i++) {
            int start = boundaries.get(i);
            int end = boundaries.get(i + 1);
            String section = content.substring(start, end).trim();
            
            if (section.isEmpty()) {
                continue;
            }
            
            // If section is too large, split it further
            if (section.length() > maxChunkSize) {
                chunks.addAll(splitLargeSection(section));
            } else if (section.length() >= minChunkSize) {
                chunks.add(section);
            } else {
                // Section is too small, try to combine with next if possible
                if (i < boundaries.size() - 2) {
                    int nextEnd = boundaries.get(i + 2);
                    String combined = content.substring(start, nextEnd).trim();
                    if (combined.length() <= maxChunkSize) {
                        chunks.add(combined);
                        i++; // Skip next section since we combined it
                        continue;
                    }
                }
                // Can't combine, add as is
                chunks.add(section);
            }
        }
        
        return chunks;
    }

    private List<String> splitLargeSection(String section) {
        List<String> chunks = new ArrayList<>();
        String[] words = section.split("\\s+");
        int wordsPerChunk = maxChunkSize / 6; // Rough estimate: 6 chars per word
        
        for (int i = 0; i < words.length; i += wordsPerChunk) {
            int end = Math.min(i + wordsPerChunk, words.length);
            String chunk = String.join(" ", java.util.Arrays.copyOfRange(words, i, end));
            chunks.add(chunk);
        }
        
        return chunks;
    }

    @Override
    public String getName() {
        return "Adaptive Chunking";
    }

    @Override
    public String getDescription() {
        return String.format(
            "Adaptive chunking based on document structure. " +
            "Detects sections using pattern and creates chunks of %d-%d characters.",
            minChunkSize, maxChunkSize
        );
    }
}

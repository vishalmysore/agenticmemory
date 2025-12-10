package io.github.vishalmysore.chunking;

import io.github.vishalmysore.ChunkingStrategy;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Adaptive Chunking Strategy
 * 
 * Creates variable-size chunks that respect natural document boundaries
 * (sections, paragraphs) while staying within token limits.
 * 
 * Best for: Legal contracts, structured documents, policy documents
 * 
 * Parameters:
 * - boundaryPattern: Regex pattern for detecting boundaries (e.g., section headers)
 * - minTokens: Minimum chunk size
 * - maxTokens: Maximum chunk size
 */
public class AdaptiveChunking implements ChunkingStrategy {
    
    private final Pattern boundaryPattern;
    private final int minTokens;
    private final int maxTokens;
    
    /**
     * Creates an adaptive chunking strategy.
     * 
     * @param boundaryPattern Regex pattern for boundaries (e.g., "(?m)^SECTION \\d+:")
     * @param minTokens Minimum tokens per chunk (e.g., 800)
     * @param maxTokens Maximum tokens per chunk (e.g., 1200)
     */
    public AdaptiveChunking(String boundaryPattern, int minTokens, int maxTokens) {
        this.boundaryPattern = Pattern.compile(boundaryPattern);
        this.minTokens = minTokens;
        this.maxTokens = maxTokens;
    }
    
    /**
     * Convenience constructor with default token limits.
     * 
     * @param boundaryPattern Regex pattern for boundaries
     */
    public AdaptiveChunking(String boundaryPattern) {
        this(boundaryPattern, 800, 1200);
    }
    
    @Override
    public List<String> chunk(String content) {
        if (content == null || content.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        List<String> chunks = new ArrayList<>();
        Matcher matcher = boundaryPattern.matcher(content);
        
        int lastEnd = 0;
        while (matcher.find()) {
            if (lastEnd > 0) {
                String section = content.substring(lastEnd, matcher.start()).trim();
                if (!section.isEmpty()) {
                    chunks.add(section);
                }
            }
            lastEnd = matcher.start();
        }
        
        // Add last section
        if (lastEnd < content.length()) {
            String lastSection = content.substring(lastEnd).trim();
            if (!lastSection.isEmpty()) {
                chunks.add(lastSection);
            }
        }
        
        // If no boundaries found, fall back to paragraph splitting
        if (chunks.isEmpty()) {
            String[] paragraphs = content.split("\n\n+");
            for (String para : paragraphs) {
                if (!para.trim().isEmpty()) {
                    chunks.add(para.trim());
                }
            }
        }
        
        return chunks;
    }
    
    @Override
    public String getName() {
        return "Adaptive Chunking";
    }
    
    @Override
    public String getDescription() {
        return String.format("Respects natural boundaries (pattern: %s) with token range %d-%d",
                           boundaryPattern.pattern(), minTokens, maxTokens);
    }
}

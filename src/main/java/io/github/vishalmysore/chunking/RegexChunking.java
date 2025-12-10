package io.github.vishalmysore.chunking;

import io.github.vishalmysore.ChunkingStrategy;

import java.util.*;
import java.util.regex.Pattern;

/**
 * Regular Expression (Regex) Chunking Strategy
 * 
 * Uses regex patterns to identify chunk boundaries in structured text.
 * Can group chunks by extracted metadata (e.g., severity levels, timestamps).
 * 
 * Best for: Log files, structured text, timestamp-based data
 * 
 * Parameters:
 * - splitPattern: Regex pattern for splitting
 * - groupByPattern: Optional pattern for grouping (e.g., severity level)
 */
public class RegexChunking implements ChunkingStrategy {
    
    private final String splitPattern;
    private final String groupByPattern;
    
    /**
     * Creates a regex chunking strategy with grouping.
     * 
     * @param splitPattern Regex pattern to split on (e.g., timestamp pattern)
     * @param groupByPattern Regex pattern to extract grouping key (e.g., severity level)
     */
    public RegexChunking(String splitPattern, String groupByPattern) {
        this.splitPattern = splitPattern;
        this.groupByPattern = groupByPattern;
    }
    
    /**
     * Creates a simple regex chunking strategy without grouping.
     * 
     * @param splitPattern Regex pattern to split on
     */
    public RegexChunking(String splitPattern) {
        this(splitPattern, null);
    }
    
    @Override
    public List<String> chunk(String content) {
        if (content == null || content.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        // Split by pattern
        String[] entries = content.split("(?=" + splitPattern + ")");
        
        if (groupByPattern == null) {
            // No grouping - return individual entries
            List<String> chunks = new ArrayList<>();
            for (String entry : entries) {
                String trimmed = entry.trim();
                if (!trimmed.isEmpty()) {
                    chunks.add(trimmed);
                }
            }
            return chunks;
        }
        
        // Group by pattern
        Map<String, List<String>> groups = new LinkedHashMap<>();
        Pattern groupPattern = Pattern.compile(groupByPattern);
        
        for (String entry : entries) {
            String trimmed = entry.trim();
            if (trimmed.isEmpty()) continue;
            
            var matcher = groupPattern.matcher(trimmed);
            if (matcher.find()) {
                String key = matcher.group(1);
                groups.computeIfAbsent(key, k -> new ArrayList<>()).add(trimmed);
            } else {
                groups.computeIfAbsent("UNCATEGORIZED", k -> new ArrayList<>()).add(trimmed);
            }
        }
        
        // Create grouped chunks
        List<String> chunks = new ArrayList<>();
        for (Map.Entry<String, List<String>> entry : groups.entrySet()) {
            String key = entry.getKey();
            String combined = "GROUP: " + key + "\n\n" + String.join("\n", entry.getValue());
            chunks.add(combined);
        }
        
        return chunks;
    }
    
    @Override
    public String getName() {
        return "Regex Chunking";
    }
    
    @Override
    public String getDescription() {
        if (groupByPattern != null) {
            return String.format("Splits on '%s', groups by '%s'", splitPattern, groupByPattern);
        }
        return "Splits on pattern: " + splitPattern;
    }
    
    public String getSplitPattern() {
        return splitPattern;
    }
    
    public String getGroupByPattern() {
        return groupByPattern;
    }
}

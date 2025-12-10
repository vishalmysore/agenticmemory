package io.github.vishalmysore.chunking;

import io.github.vishalmysore.ChunkingStrategy;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * HTML/XML Tag-Based Chunking Strategy
 * 
 * Uses HTML/XML document structure to determine chunk boundaries.
 * Preserves hierarchical relationships while creating semantically complete chunks.
 * 
 * Best for: Web content, HTML documentation, XML documents
 * 
 * Parameters:
 * - splitTag: HTML tag to use as boundary (e.g., "h2", "section", "article")
 */
public class HTMLTagBasedChunking implements ChunkingStrategy {
    
    private final String splitTag;
    private final boolean stripTags;
    
    /**
     * Creates an HTML tag-based chunking strategy.
     * 
     * @param splitTag The HTML tag to split on (e.g., "h2", "section")
     * @param stripTags Whether to remove HTML tags from the output
     */
    public HTMLTagBasedChunking(String splitTag, boolean stripTags) {
        this.splitTag = splitTag;
        this.stripTags = stripTags;
    }
    
    /**
     * Convenience constructor that strips tags by default.
     * 
     * @param splitTag The HTML tag to split on
     */
    public HTMLTagBasedChunking(String splitTag) {
        this(splitTag, true);
    }
    
    @Override
    public List<String> chunk(String content) {
        if (content == null || content.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        List<String> chunks = new ArrayList<>();
        Pattern sectionPattern = Pattern.compile("<" + splitTag + ">([^<]+)</" + splitTag + ">");
        Matcher matcher = sectionPattern.matcher(content);
        
        int lastEnd = 0;
        while (matcher.find()) {
            if (lastEnd > 0) {
                String section = content.substring(lastEnd, matcher.start());
                String processed = stripTags ? section.replaceAll("<[^>]+>", "").trim() : section.trim();
                if (!processed.isEmpty()) {
                    chunks.add(processed);
                }
            }
            lastEnd = matcher.start();
        }
        
        // Add last section
        if (lastEnd < content.length()) {
            String section = content.substring(lastEnd);
            String processed = stripTags ? section.replaceAll("<[^>]+>", "").trim() : section.trim();
            if (!processed.isEmpty()) {
                chunks.add(processed);
            }
        }
        
        // Fallback: if no tags found, return cleaned content
        if (chunks.isEmpty()) {
            String processed = stripTags ? content.replaceAll("<[^>]+>", "").trim() : content.trim();
            if (!processed.isEmpty()) {
                chunks.add(processed);
            }
        }
        
        return chunks;
    }
    
    @Override
    public String getName() {
        return "HTML Tag-Based Chunking";
    }
    
    @Override
    public String getDescription() {
        return String.format("Splits on <%s> tags, %s HTML markup", 
                           splitTag, stripTags ? "removing" : "preserving");
    }
    
    public String getSplitTag() {
        return splitTag;
    }
}

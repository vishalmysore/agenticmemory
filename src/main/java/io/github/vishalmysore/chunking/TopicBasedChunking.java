package io.github.vishalmysore.chunking;

import io.github.vishalmysore.ChunkingStrategy;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Topic/Theme-Based Chunking Strategy
 * 
 * Groups content by underlying topics or themes using section markers.
 * Aggregates similar content types across different documents/sections.
 * 
 * Best for: Research papers, technical documentation, structured biographies
 * 
 * Parameters:
 * - topicPattern: Regex pattern to identify topic markers
 */
public class TopicBasedChunking implements ChunkingStrategy {
    
    private final Pattern topicPattern;
    private final Pattern personPattern;
    
    /**
     * Creates a topic-based chunking strategy.
     * 
     * @param topicPattern Regex pattern for topic markers (e.g., "(EDUCATION|CAREER|PATENTS):")
     */
    public TopicBasedChunking(String topicPattern) {
        this.topicPattern = Pattern.compile(topicPattern);
        this.personPattern = Pattern.compile("(?m)^===\\s*$");
    }
    
    @Override
    public List<String> chunk(String content) {
        if (content == null || content.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        // Group by topics across all sections
        Map<String, List<String>> topicChunks = new LinkedHashMap<>();
        String[] sections = personPattern.split(content);
        
        for (String section : sections) {
            if (section.trim().isEmpty()) continue;
            
            String personName = extractPersonName(section);
            Matcher matcher = topicPattern.matcher(section);
            
            int lastPos = 0;
            String currentTopic = "";
            
            while (matcher.find()) {
                if (lastPos > 0 && !currentTopic.isEmpty()) {
                    String sectionContent = section.substring(lastPos, matcher.start()).trim();
                    topicChunks.computeIfAbsent(currentTopic, k -> new ArrayList<>())
                              .add(personName + ": " + sectionContent);
                }
                currentTopic = cleanTopicName(matcher.group(1));
                lastPos = matcher.start();
            }
            
            // Add last topic
            if (lastPos > 0 && !currentTopic.isEmpty() && lastPos < section.length()) {
                String sectionContent = section.substring(lastPos).trim();
                if (!sectionContent.isEmpty()) {
                    topicChunks.computeIfAbsent(currentTopic, k -> new ArrayList<>())
                              .add(personName + ": " + sectionContent);
                }
            }
        }
        
        // Create topic-based chunks
        List<String> chunks = new ArrayList<>();
        for (Map.Entry<String, List<String>> entry : topicChunks.entrySet()) {
            String topic = entry.getKey();
            String chunk = "TOPIC: " + topic + "\n\n" + String.join("\n\n", entry.getValue());
            chunks.add(chunk);
        }
        
        return chunks;
    }
    
    private String extractPersonName(String section) {
        String[] lines = section.split("\n");
        for (String line : lines) {
            if (line.contains("PERSON:") || line.contains("👤")) {
                return line.replace("👤", "").replace("PERSON:", "").trim();
            }
        }
        return lines[0].trim().substring(0, Math.min(30, lines[0].trim().length()));
    }
    
    private String cleanTopicName(String topic) {
        return topic.replace(":", "")
                   .replace("🎓", "").replace("💼", "")
                   .replace("🧑‍💻", "").replace("📝", "")
                   .replace("🧪", "").replace("🔎", "")
                   .trim();
    }
    
    @Override
    public String getName() {
        return "Topic/Theme-Based Chunking";
    }
    
    @Override
    public String getDescription() {
        return "Groups content by topics/themes using pattern: " + topicPattern.pattern();
    }
}

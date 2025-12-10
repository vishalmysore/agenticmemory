package io.github.vishalmysore.chunking;

import io.github.vishalmysore.ChunkingStrategy;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Task-Aware Chunking Strategy
 * 
 * Adapts chunking based on the downstream task (summarization, search, Q&A).
 * Different tasks benefit from different chunk sizes and boundaries.
 * 
 * Best for: Code repositories, multi-purpose document processing
 * 
 * Parameters:
 * - taskType: The task for which to optimize chunking
 */
public class TaskAwareChunking implements ChunkingStrategy {
    
    public enum TaskType {
        SUMMARIZATION,  // Small chunks (function-level)
        SEARCH,         // Medium chunks (signatures + docstrings)
        QA              // Large chunks (full classes/modules)
    }
    
    private final TaskType taskType;
    
    /**
     * Creates a task-aware chunking strategy.
     * 
     * @param taskType The task type to optimize for
     */
    public TaskAwareChunking(TaskType taskType) {
        this.taskType = taskType;
    }
    
    @Override
    public List<String> chunk(String content) {
        if (content == null || content.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        switch (taskType) {
            case SUMMARIZATION:
                return chunkForSummarization(content);
            case SEARCH:
                return chunkForSearch(content);
            case QA:
                return chunkForQA(content);
            default:
                return List.of(content);
        }
    }
    
    private List<String> chunkForSummarization(String content) {
        // Small function-level chunks
        List<String> chunks = new ArrayList<>();
        Pattern functionPattern = Pattern.compile("(?m)^def \\w+\\([^)]*\\):");
        Matcher matcher = functionPattern.matcher(content);
        
        while (matcher.find()) {
            int start = matcher.start();
            int end = content.indexOf("\n\ndef ", start + 1);
            if (end == -1) end = content.indexOf("\n\nif __name__", start + 1);
            if (end == -1) end = content.indexOf("\n\nclass ", start + 1);
            if (end == -1) end = content.length();
            
            String function = content.substring(start, end).trim();
            chunks.add(function);
        }
        
        return chunks.isEmpty() ? List.of(content) : chunks;
    }
    
    private List<String> chunkForSearch(String content) {
        // Signatures + docstrings only
        List<String> chunks = new ArrayList<>();
        Pattern docstringPattern = Pattern.compile("(?m)(def \\w+\\([^)]*\\):\n\\s+\"\"\"[^\"]+\"\"\")");
        Matcher matcher = docstringPattern.matcher(content);
        
        while (matcher.find()) {
            chunks.add(matcher.group(1));
        }
        
        return chunks.isEmpty() ? List.of(content) : chunks;
    }
    
    private List<String> chunkForQA(String content) {
        // Full classes for context
        List<String> chunks = new ArrayList<>();
        Pattern classPattern = Pattern.compile("(?m)^class \\w+:");
        Matcher matcher = classPattern.matcher(content);
        
        int lastEnd = 0;
        while (matcher.find()) {
            int start = matcher.start();
            int end = content.indexOf("\n\nclass ", start + 1);
            if (end == -1) end = content.indexOf("\n\ndef ", start + 1);
            if (end == -1) end = content.length();
            
            String classCode = content.substring(start, end).trim();
            chunks.add(classCode);
            lastEnd = end;
        }
        
        return chunks.isEmpty() ? List.of(content) : chunks;
    }
    
    @Override
    public String getName() {
        return "Task-Aware Chunking";
    }
    
    @Override
    public String getDescription() {
        return "Optimized for " + taskType + " task";
    }
    
    public TaskType getTaskType() {
        return taskType;
    }
}

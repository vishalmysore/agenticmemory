package io.github.vishalmysore.chunking;

import io.github.vishalmysore.ChunkingStrategy;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Code-Specific Chunking Strategy
 * 
 * Uses programming language syntax to create meaningful code chunks.
 * Splits at logical boundaries (classes, functions, methods) inspired by AST analysis.
 * 
 * Best for: Source code, code repositories, programming tutorials
 * 
 * Parameters:
 * - language: Programming language (PYTHON, JAVA, JAVASCRIPT)
 */
public class CodeSpecificChunking implements ChunkingStrategy {
    
    public enum Language {
        PYTHON,
        JAVA,
        JAVASCRIPT
    }
    
    private final Language language;
    
    /**
     * Creates a code-specific chunking strategy.
     * 
     * @param language The programming language to parse
     */
    public CodeSpecificChunking(Language language) {
        this.language = language;
    }
    
    @Override
    public List<String> chunk(String content) {
        if (content == null || content.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        switch (language) {
            case PYTHON:
                return chunkPython(content);
            case JAVA:
                return chunkJava(content);
            case JAVASCRIPT:
                return chunkJavaScript(content);
            default:
                return List.of(content);
        }
    }
    
    private List<String> chunkPython(String content) {
        List<String> chunks = new ArrayList<>();
        
        // Find all classes
        Pattern classPattern = Pattern.compile("(?m)^class (\\w+):");
        Matcher classMatcher = classPattern.matcher(content);
        
        while (classMatcher.find()) {
            int start = classMatcher.start();
            int end = content.indexOf("\n\nclass ", start + 1);
            if (end == -1) end = content.indexOf("\n\ndef ", start + 1);
            if (end == -1) end = content.length();
            
            String classCode = content.substring(start, end).trim();
            String className = classMatcher.group(1);
            chunks.add("# Class: " + className + "\n" + classCode);
        }
        
        // Find standalone functions (not indented = not methods)
        Pattern funcPattern = Pattern.compile("(?m)^def (\\w+)\\(");
        Matcher funcMatcher = funcPattern.matcher(content);
        
        while (funcMatcher.find()) {
            int start = funcMatcher.start();
            int end = content.indexOf("\n\ndef ", start + 1);
            if (end == -1) end = content.indexOf("\n\nif __name__", start + 1);
            if (end == -1) end = content.length();
            
            String funcCode = content.substring(start, end).trim();
            String funcName = funcMatcher.group(1);
            chunks.add("# Function: " + funcName + "\n" + funcCode);
        }
        
        return chunks.isEmpty() ? List.of(content) : chunks;
    }
    
    private List<String> chunkJava(String content) {
        List<String> chunks = new ArrayList<>();
        
        // Find all classes/interfaces
        Pattern classPattern = Pattern.compile("(?m)^(public |private |protected )?(class|interface) (\\w+)");
        Matcher matcher = classPattern.matcher(content);
        
        while (matcher.find()) {
            int start = matcher.start();
            // Find matching closing brace (simplified)
            int braceCount = 0;
            int end = start;
            boolean inClass = false;
            
            for (int i = start; i < content.length(); i++) {
                if (content.charAt(i) == '{') {
                    braceCount++;
                    inClass = true;
                } else if (content.charAt(i) == '}') {
                    braceCount--;
                    if (inClass && braceCount == 0) {
                        end = i + 1;
                        break;
                    }
                }
            }
            
            if (end > start) {
                String classCode = content.substring(start, end).trim();
                String className = matcher.group(3);
                chunks.add("// " + matcher.group(2) + ": " + className + "\n" + classCode);
            }
        }
        
        return chunks.isEmpty() ? List.of(content) : chunks;
    }
    
    private List<String> chunkJavaScript(String content) {
        List<String> chunks = new ArrayList<>();
        
        // Find all function declarations and classes
        Pattern pattern = Pattern.compile("(?m)^(function \\w+\\(|class \\w+ |const \\w+ = \\([^)]*\\) =>)");
        Matcher matcher = pattern.matcher(content);
        
        int lastEnd = 0;
        while (matcher.find()) {
            if (lastEnd > 0) {
                chunks.add(content.substring(lastEnd, matcher.start()).trim());
            }
            lastEnd = matcher.start();
        }
        
        if (lastEnd < content.length()) {
            chunks.add(content.substring(lastEnd).trim());
        }
        
        return chunks.isEmpty() ? List.of(content) : chunks;
    }
    
    @Override
    public String getName() {
        return "Code-Specific Chunking";
    }
    
    @Override
    public String getDescription() {
        return "AST-inspired chunking for " + language + " code (classes, functions, methods)";
    }
    
    public Language getLanguage() {
        return language;
    }
}

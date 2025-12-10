package io.github.vishalmysore.chunking;

import io.github.vishalmysore.ChunkingStrategy;

import java.util.ArrayList;
import java.util.List;

/**
 * Hybrid Chunking Strategy
 * 
 * Combines multiple chunking strategies in a pipeline.
 * Each stage processes the output of the previous stage.
 * 
 * Best for: Complex documents requiring multi-stage processing
 * 
 * Parameters:
 * - strategies: Array of strategies to apply in sequence
 */
public class HybridChunking implements ChunkingStrategy {
    
    private final ChunkingStrategy[] strategies;
    
    /**
     * Creates a hybrid chunking strategy that applies strategies in sequence.
     * 
     * @param strategies Array of strategies to apply (in order)
     */
    public HybridChunking(ChunkingStrategy... strategies) {
        if (strategies == null || strategies.length == 0) {
            throw new IllegalArgumentException("At least one strategy must be provided");
        }
        this.strategies = strategies;
    }
    
    @Override
    public List<String> chunk(String content) {
        if (content == null || content.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        // Start with the original content as a single chunk
        List<String> currentChunks = new ArrayList<>();
        currentChunks.add(content);
        
        // Apply each strategy in the pipeline
        for (ChunkingStrategy strategy : strategies) {
            List<String> nextChunks = new ArrayList<>();
            
            for (String chunk : currentChunks) {
                List<String> subChunks = strategy.chunk(chunk);
                nextChunks.addAll(subChunks);
            }
            
            currentChunks = nextChunks;
        }
        
        return currentChunks;
    }
    
    @Override
    public String getName() {
        return "Hybrid Chunking";
    }
    
    @Override
    public String getDescription() {
        StringBuilder sb = new StringBuilder("Pipeline of " + strategies.length + " strategies: ");
        for (int i = 0; i < strategies.length; i++) {
            if (i > 0) sb.append(" → ");
            sb.append(strategies[i].getName());
        }
        return sb.toString();
    }
    
    public ChunkingStrategy[] getStrategies() {
        return strategies;
    }
}

package io.github.vishalmysore.chunking;

import io.github.vishalmysore.ChunkingStrategy;

import java.util.*;

/**
 * Entity-Based Chunking Strategy
 * 
 * Groups sentences by the entities (people, companies, locations) they mention.
 * Useful for documents with multiple subjects where entity-centric retrieval is important.
 * 
 * Best for: News articles, research papers, multi-person biographies
 * 
 * Parameters:
 * - entities: Array of entity names to detect and group by
 */
public class EntityBasedChunking implements ChunkingStrategy {
    
    private final String[] entities;
    
    /**
     * Creates an entity-based chunking strategy.
     * 
     * @param entities Array of entity names to detect (e.g., ["Elon Musk", "Tesla", "SpaceX"])
     */
    public EntityBasedChunking(String... entities) {
        if (entities == null || entities.length == 0) {
            throw new IllegalArgumentException("At least one entity must be provided");
        }
        this.entities = entities;
    }
    
    @Override
    public List<String> chunk(String content) {
        if (content == null || content.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        // Split into sentences
        String[] sentences = content.split("\\. ");
        
        // Group sentences by entity
        Map<String, List<String>> entityChunks = new LinkedHashMap<>();
        
        for (String sentence : sentences) {
            for (String entity : entities) {
                if (sentence.contains(entity)) {
                    entityChunks.computeIfAbsent(entity, k -> new ArrayList<>())
                               .add(sentence.trim() + ".");
                }
            }
        }
        
        // Create chunks with entity labels
        List<String> chunks = new ArrayList<>();
        for (Map.Entry<String, List<String>> entry : entityChunks.entrySet()) {
            String entity = entry.getKey();
            String chunk = "ENTITY: " + entity + "\n\n" + String.join(" ", entry.getValue());
            chunks.add(chunk);
        }
        
        return chunks;
    }
    
    @Override
    public String getName() {
        return "Entity-Based Chunking";
    }
    
    @Override
    public String getDescription() {
        return String.format("Groups sentences by %d entities: %s", 
                           entities.length, String.join(", ", Arrays.copyOf(entities, Math.min(3, entities.length))));
    }
    
    public String[] getEntities() {
        return entities;
    }
}

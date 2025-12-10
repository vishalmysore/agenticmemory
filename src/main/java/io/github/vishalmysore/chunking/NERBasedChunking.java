package io.github.vishalmysore.chunking;

import io.github.vishalmysore.ChunkingStrategy;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.util.Span;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.*;

/**
 * NER-Based Chunking Strategy using Apache OpenNLP
 * 
 * Automatically detects named entities (PERSON, ORGANIZATION, LOCATION) 
 * without requiring a predefined list. Groups sentences by the entities they mention.
 * 
 * This is more advanced than EntityBasedChunking as it:
 * - Automatically discovers entities in the text
 * - Uses machine learning models for entity recognition
 * - Handles variations and contexts better
 * 
 * Best for: News articles, research papers, documents with unknown entities
 * 
 * Requirements:
 * - Apache OpenNLP models must be downloaded from https://opennlp.apache.org/models.html
 * - Required models: en-ner-person.bin, en-ner-organization.bin, en-ner-location.bin
 * 
 * Example usage:
 * <pre>
 * ChunkingStrategy strategy = new NERBasedChunking(
 *     "models/en-ner-person.bin",
 *     "models/en-ner-organization.bin", 
 *     "models/en-ner-location.bin"
 * );
 * </pre>
 */
public class NERBasedChunking implements ChunkingStrategy {
    
    private final NameFinderME personFinder;
    private final NameFinderME organizationFinder;
    private final NameFinderME locationFinder;
    private final SimpleTokenizer tokenizer;
    private final boolean caseInsensitive;
    
    /**
     * Creates NER-based chunking strategy with OpenNLP models.
     * 
     * @param personModelPath Path to en-ner-person.bin model file
     * @param orgModelPath Path to en-ner-organization.bin model file
     * @param locationModelPath Path to en-ner-location.bin model file
     * @throws Exception if model files cannot be loaded
     */
    public NERBasedChunking(String personModelPath, String orgModelPath, String locationModelPath) 
            throws Exception {
        this(personModelPath, orgModelPath, locationModelPath, true);
    }
    
    /**
     * Creates NER-based chunking strategy with OpenNLP models.
     * 
     * @param personModelPath Path to en-ner-person.bin model file
     * @param orgModelPath Path to en-ner-organization.bin model file
     * @param locationModelPath Path to en-ner-location.bin model file
     * @param caseInsensitive Whether to use case-insensitive matching for grouping
     * @throws Exception if model files cannot be loaded
     */
    public NERBasedChunking(String personModelPath, String orgModelPath, String locationModelPath,
                           boolean caseInsensitive) throws Exception {
        
        try (InputStream personModelIn = new FileInputStream(personModelPath);
             InputStream orgModelIn = new FileInputStream(orgModelPath);
             InputStream locationModelIn = new FileInputStream(locationModelPath)) {
            
            TokenNameFinderModel personModel = new TokenNameFinderModel(personModelIn);
            TokenNameFinderModel orgModel = new TokenNameFinderModel(orgModelIn);
            TokenNameFinderModel locationModel = new TokenNameFinderModel(locationModelIn);
            
            this.personFinder = new NameFinderME(personModel);
            this.organizationFinder = new NameFinderME(orgModel);
            this.locationFinder = new NameFinderME(locationModel);
            this.tokenizer = SimpleTokenizer.INSTANCE;
            this.caseInsensitive = caseInsensitive;
        }
    }
    
    @Override
    public List<String> chunk(String content) {
        if (content == null || content.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        // Split into sentences
        String[] sentences = content.split("\\. ");
        
        // Detect entities in the entire document first
        Set<String> allEntities = new LinkedHashSet<>();
        
        for (String sentence : sentences) {
            String[] tokens = tokenizer.tokenize(sentence);
            
            // Find persons
            Span[] personSpans = personFinder.find(tokens);
            for (Span span : personSpans) {
                String entity = String.join(" ", 
                    Arrays.copyOfRange(tokens, span.getStart(), span.getEnd()));
                allEntities.add(entity);
            }
            personFinder.clearAdaptiveData();
            
            // Find organizations
            Span[] orgSpans = organizationFinder.find(tokens);
            for (Span span : orgSpans) {
                String entity = String.join(" ", 
                    Arrays.copyOfRange(tokens, span.getStart(), span.getEnd()));
                allEntities.add(entity);
            }
            organizationFinder.clearAdaptiveData();
            
            // Find locations
            Span[] locationSpans = locationFinder.find(tokens);
            for (Span span : locationSpans) {
                String entity = String.join(" ", 
                    Arrays.copyOfRange(tokens, span.getStart(), span.getEnd()));
                allEntities.add(entity);
            }
            locationFinder.clearAdaptiveData();
        }
        
        // Group sentences by entity
        Map<String, List<String>> entityChunks = new LinkedHashMap<>();
        
        for (String sentence : sentences) {
            for (String entity : allEntities) {
                boolean matches = caseInsensitive 
                    ? sentence.toLowerCase().contains(entity.toLowerCase())
                    : sentence.contains(entity);
                    
                if (matches) {
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
        return "NER-Based Chunking (OpenNLP)";
    }
    
    @Override
    public String getDescription() {
        return "Automatically detects entities (PERSON, ORGANIZATION, LOCATION) using Apache OpenNLP NER models";
    }
    
    /**
     * Gets the detected entities from the last chunk operation.
     * Note: This is a stateless operation, call chunk() to detect entities.
     * 
     * @param content The content to analyze
     * @return Set of detected entity names
     */
    public Set<String> detectEntities(String content) {
        if (content == null || content.trim().isEmpty()) {
            return new HashSet<>();
        }
        
        String[] sentences = content.split("\\. ");
        Set<String> entities = new LinkedHashSet<>();
        
        for (String sentence : sentences) {
            String[] tokens = tokenizer.tokenize(sentence);
            
            // Find persons
            Span[] personSpans = personFinder.find(tokens);
            for (Span span : personSpans) {
                String entity = String.join(" ", 
                    Arrays.copyOfRange(tokens, span.getStart(), span.getEnd()));
                entities.add(entity + " (PERSON)");
            }
            personFinder.clearAdaptiveData();
            
            // Find organizations
            Span[] orgSpans = organizationFinder.find(tokens);
            for (Span span : orgSpans) {
                String entity = String.join(" ", 
                    Arrays.copyOfRange(tokens, span.getStart(), span.getEnd()));
                entities.add(entity + " (ORGANIZATION)");
            }
            organizationFinder.clearAdaptiveData();
            
            // Find locations
            Span[] locationSpans = locationFinder.find(tokens);
            for (Span span : locationSpans) {
                String entity = String.join(" ", 
                    Arrays.copyOfRange(tokens, span.getStart(), span.getEnd()));
                entities.add(entity + " (LOCATION)");
            }
            locationFinder.clearAdaptiveData();
        }
        
        return entities;
    }
}

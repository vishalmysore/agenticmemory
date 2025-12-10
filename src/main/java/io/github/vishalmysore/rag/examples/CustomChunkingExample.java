package io.github.vishalmysore.rag.examples;

import io.github.vishalmysore.ChunkingStrategy;
import io.github.vishalmysore.OpenAIEmbeddingProvider;
import io.github.vishalmysore.RAGService;
import io.github.vishalmysore.SearchResult;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Example demonstrating how to create a custom chunking strategy.
 * 
 * This shows the extensibility of the framework - anyone can implement
 * ChunkingStrategy to create their own chunking logic.
 */
public class CustomChunkingExample {

    /**
     * Custom chunking strategy that splits on sentence boundaries
     * and groups every N sentences together.
     */
    public static class SentenceGroupChunking implements ChunkingStrategy {
        
        private final int sentencesPerChunk;
        
        public SentenceGroupChunking(int sentencesPerChunk) {
            this.sentencesPerChunk = sentencesPerChunk;
        }
        
        @Override
        public List<String> chunk(String content) {
            if (content == null || content.trim().isEmpty()) {
                return new ArrayList<>();
            }
            
            // Split by sentence endings
            String[] sentences = content.split("(?<=[.!?])\\s+");
            List<String> chunks = new ArrayList<>();
            
            for (int i = 0; i < sentences.length; i += sentencesPerChunk) {
                int end = Math.min(i + sentencesPerChunk, sentences.length);
                StringBuilder chunk = new StringBuilder();
                
                for (int j = i; j < end; j++) {
                    if (j > i) chunk.append(" ");
                    chunk.append(sentences[j]);
                }
                
                chunks.add(chunk.toString());
            }
            
            return chunks;
        }
        
        @Override
        public String getName() {
            return "Sentence Group Chunking";
        }
        
        @Override
        public String getDescription() {
            return "Groups every " + sentencesPerChunk + " sentences into a chunk";
        }
    }

    public static void main(String[] args) throws Exception {
        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null || apiKey.isEmpty()) {
            System.err.println("Error: OPENAI_API_KEY environment variable is required!");
            return;
        }

        System.out.println("=== Custom Chunking Strategy Example ===\n");

        // Create our custom strategy
        ChunkingStrategy customStrategy = new SentenceGroupChunking(3);
        
        System.out.println("Strategy: " + customStrategy.getName());
        System.out.println("Description: " + customStrategy.getDescription());
        System.out.println();

        String sampleText = """
            Artificial Intelligence is transforming the world. Machine learning algorithms 
            can now process vast amounts of data. Neural networks are becoming increasingly 
            sophisticated. Deep learning has enabled breakthroughs in image recognition. 
            Natural language processing allows computers to understand human language. 
            Generative AI can create original content. The future of AI is incredibly exciting. 
            Ethical considerations are paramount as AI advances.
            """;

        OpenAIEmbeddingProvider embeddings = new OpenAIEmbeddingProvider(apiKey, "text-embedding-3-small", 1024);

        try (RAGService rag = new RAGService(Paths.get("custom-chunking-index"), embeddings)) {
            // Use our custom chunking strategy
            int chunkCount = rag.addDocumentWithChunking("ai_article", sampleText, customStrategy);
            rag.commit();
            
            System.out.println("✓ Created " + chunkCount + " chunks using custom strategy\n");
            
            // Query the indexed content
            System.out.println("Query: \"What can neural networks do?\"");
            List<SearchResult> results = rag.search("What can neural networks do?", 2);
            
            for (int i = 0; i < results.size(); i++) {
                System.out.printf("\nResult %d (%.1f%% match):%n", i + 1, results.get(i).getScore() * 100);
                System.out.println(results.get(i).getContent());
            }
        } finally {
            embeddings.close();
        }
    }
}

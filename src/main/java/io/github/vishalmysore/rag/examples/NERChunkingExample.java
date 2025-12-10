package io.github.vishalmysore.rag.examples;

import io.github.vishalmysore.OpenAIEmbeddingProvider;
import io.github.vishalmysore.RAGService;
import io.github.vishalmysore.SearchResult;
import io.github.vishalmysore.chunking.NERBasedChunking;

import java.nio.file.Paths;
import java.util.List;
import java.util.Set;

/**
 * Example demonstrating NER-Based Chunking with Apache OpenNLP.
 * 
 * This example shows how to use automatic Named Entity Recognition to:
 * 1. Detect entities (people, organizations, locations) in text
 * 2. Group sentences by mentioned entities
 * 3. Enable entity-centric retrieval
 * 
 * Prerequisites:
 * - Download OpenNLP models from https://opennlp.apache.org/models.html
 * - Place models in src/main/resources/opennlp-models/ directory:
 *   - en-ner-person.bin
 *   - en-ner-organization.bin
 *   - en-ner-location.bin
 * 
 * Usage:
 * 1. Download models (see above)
 * 2. Set OPENAI_API_KEY environment variable
 * 3. Run: mvn exec:java -Dexec.mainClass="io.github.vishalmysore.rag.examples.NERChunkingExample" -q
 */
public class NERChunkingExample {

    public static void main(String[] args) throws Exception {
        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null || apiKey.isEmpty()) {
            System.err.println("Error: OPENAI_API_KEY environment variable is required!");
            return;
        }

        System.out.println("=== NER-Based Chunking Example ===\n");

        // Sample news article with multiple entities
        String newsArticle = """
            TECH NEWS DIGEST - January 15, 2025
            
            Elon Musk announced a groundbreaking new AI initiative at Tesla's headquarters in Palo Alto.
            The project aims to integrate advanced neural networks into Tesla's Autopilot system.
            According to sources at SpaceX, Musk is also exploring similar technology for spacecraft navigation.
            
            Meanwhile, Mark Zuckerberg revealed Meta's latest virtual reality advancements at the company's 
            Menlo Park campus. The new Quest 4 headset features unprecedented processing power developed 
            in collaboration with Qualcomm.
            
            In other news, Tim Cook spoke at Apple's Cupertino headquarters about the company's commitment 
            to sustainable manufacturing. Apple plans to achieve complete carbon neutrality by 2030.
            
            Jeff Bezos, through Blue Origin, announced plans for a new lunar mission scheduled for late 2026.
            The mission will launch from Cape Canaveral and aims to establish a permanent research base.
            
            Google's Sundar Pichai discussed the company's AI research at the Mountain View campus,
            highlighting breakthroughs in natural language processing and quantum computing initiatives.
            """;

        // Paths to OpenNLP models
        String personModel = "src/main/resources/opennlp-models/en-ner-person.bin";
        String orgModel = "src/main/resources/opennlp-models/en-ner-organization.bin";
        String locationModel = "src/main/resources/opennlp-models/en-ner-location.bin";

        try {
            // Create NER-based chunking strategy
            NERBasedChunking nerStrategy = new NERBasedChunking(personModel, orgModel, locationModel);
            
            System.out.println("Strategy: " + nerStrategy.getName());
            System.out.println("Description: " + nerStrategy.getDescription());
            System.out.println();

            // First, detect entities to show what was found
            System.out.println("Detecting entities in the article...\n");
            Set<String> detectedEntities = nerStrategy.detectEntities(newsArticle);
            
            System.out.println("Detected " + detectedEntities.size() + " entities:");
            for (String entity : detectedEntities) {
                System.out.println("  • " + entity);
            }
            System.out.println();

            // Now use it with RAG
            OpenAIEmbeddingProvider embeddings = new OpenAIEmbeddingProvider(
                apiKey, "text-embedding-3-small", 1024
            );

            try (RAGService rag = new RAGService(Paths.get("ner-chunking-index"), embeddings)) {
                // Add document with automatic NER chunking
                int chunkCount = rag.addDocumentWithChunking("tech_news", newsArticle, nerStrategy);
                rag.commit();
                
                System.out.println("✓ Created " + chunkCount + " entity-based chunks using NER\n");
                
                // Query the indexed content
                System.out.println("=== Entity-Centric Queries ===\n");
                
                queryAndDisplay(rag, "What did Elon Musk announce?");
                queryAndDisplay(rag, "Tell me about Meta's developments");
                queryAndDisplay(rag, "What is happening at Apple?");
                queryAndDisplay(rag, "What are the space exploration plans?");
                
            } finally {
                embeddings.close();
            }

            System.out.println("\n✓ Demo complete!");
            System.out.println("\nNote: NER automatically detected entities without manual configuration!");
            System.out.println("Compare this to EntityBasedChunking which requires predefined entity list.");

        } catch (Exception e) {
            System.err.println("\nError: " + e.getMessage());
            System.err.println("\nMake sure you have downloaded OpenNLP models to:");
            System.err.println("  - src/main/resources/opennlp-models/en-ner-person.bin");
            System.err.println("  - src/main/resources/opennlp-models/en-ner-organization.bin");
            System.err.println("  - src/main/resources/opennlp-models/en-ner-location.bin");
            System.err.println("\nDownload from: https://opennlp.apache.org/models.html");
            throw e;
        }
    }

    private static void queryAndDisplay(RAGService rag, String question) throws Exception {
        System.out.println("Q: " + question);
        List<SearchResult> results = rag.search(question, 2);
        
        if (results.isEmpty()) {
            System.out.println("   No results found\n");
            return;
        }

        SearchResult top = results.get(0);
        String preview = top.getContent().length() > 150 
            ? top.getContent().substring(0, 150) + "..." 
            : top.getContent();
            
        System.out.printf("   Answer (%.1f%% match): %s%n%n", top.getScore() * 100, preview);
    }
}

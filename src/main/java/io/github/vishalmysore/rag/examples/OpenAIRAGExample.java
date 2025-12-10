package io.github.vishalmysore.rag.examples;

import io.github.vishalmysore.*;

import java.nio.file.Paths;
import java.util.List;

/**
 * Example demonstrating how to use the OpenAI embedding provider with RAGService.
 * 
 * To run this example, you need:
 * 1. An OpenAI API key
 * 2. Set the OPENAI_API_KEY environment variable or replace "your-api-key-here" with your actual key
 */
public class OpenAIRAGExample {

    public static void main(String[] args) throws Exception {
        // Get API key from environment variable (recommended) or hardcode it
        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null || apiKey.isEmpty()) {
            System.err.println("Please set OPENAI_API_KEY environment variable");
            System.err.println("Or modify the code to include your API key directly");
            return;
        }

        System.out.println("=== OpenAI RAG Example ===\n");

        // Create OpenAI embedding provider
        // Using text-embedding-3-small model with 1024 dimensions (Lucene max)
        OpenAIEmbeddingProvider embeddings = new OpenAIEmbeddingProvider(
            apiKey,
            "text-embedding-3-small",
            1024
        );

        try {
            // Initialize RAG service with OpenAI embeddings
            try (RAGService rag = new RAGService(Paths.get("openai-rag-index"), embeddings)) {
                
                // Add documents about AI and technology
                System.out.println("Adding documents...");
                
                rag.addDocument(
                    "doc1",
                    "Artificial intelligence is the simulation of human intelligence by machines. " +
                    "Machine learning is a subset of AI that allows systems to learn from data."
                );
                
                rag.addDocument(
                    "doc2",
                    "Deep learning uses neural networks with multiple layers to process data. " +
                    "It excels at tasks like image recognition and natural language processing."
                );
                
                rag.addDocument(
                    "doc3",
                    "Python is a high-level programming language widely used for AI and data science. " +
                    "It has extensive libraries like TensorFlow, PyTorch, and scikit-learn."
                );
                
                rag.addDocument(
                    "doc4",
                    "Natural language processing enables computers to understand human language. " +
                    "Applications include chatbots, translation, and sentiment analysis."
                );
                
                rag.addDocument(
                    "doc5",
                    "Java is a robust, object-oriented programming language used in enterprise applications. " +
                    "It runs on the Java Virtual Machine, enabling cross-platform compatibility."
                );

                // Commit the changes
                rag.commit();
                System.out.println("Documents indexed successfully!\n");

                // Perform semantic searches
                performSearch(rag, "What is AI and machine learning?", 3);
                performSearch(rag, "Tell me about neural networks", 2);
                performSearch(rag, "Which programming languages are good for AI?", 3);

                // Demonstrate context retrieval for RAG
                System.out.println("\n=== Context Retrieval for RAG ===");
                String context = rag.retrieveContext("What is deep learning?", 2);
                System.out.println("Retrieved Context:\n" + context);
                System.out.println("\n(This context can be passed to an LLM for question answering)\n");

                // Demonstrate hybrid search
                System.out.println("=== Hybrid Search (Vector + Keyword) ===");
                System.out.println("Query: 'programming language'");
                System.out.println("Weight: 0.7 (70% semantic, 30% keyword)\n");
                
                List<SearchResult> hybridResults = rag.hybridSearch("programming language", 3, 0.7f);
                for (int i = 0; i < hybridResults.size(); i++) {
                    SearchResult result = hybridResults.get(i);
                    System.out.printf("%d. [Score: %.4f] %s%n", 
                        i + 1, result.getScore(), result.getContent());
                }

                System.out.println("\n=== Example Complete ===");
                System.out.println("✓ Used OpenAI embeddings for semantic search");
                System.out.println("✓ All data stored locally in Lucene index");
                System.out.println("✓ No vector database server required!");
            }
        } finally {
            // Clean up resources
            embeddings.close();
            System.out.println("\nResources cleaned up successfully.");
        }
    }

    private static void performSearch(RAGService rag, String query, int topK) throws Exception {
        System.out.println("=== Search Query ===");
        System.out.println("Query: " + query);
        System.out.println("Top " + topK + " results:\n");

        List<SearchResult> results = rag.search(query, topK);
        
        for (int i = 0; i < results.size(); i++) {
            SearchResult result = results.get(i);
            System.out.printf("%d. [Score: %.4f] %s%n%n", 
                i + 1, result.getScore(), result.getContent());
        }
    }
}

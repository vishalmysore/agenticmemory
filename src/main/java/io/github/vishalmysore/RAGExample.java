package io.github.vishalmysore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Example usage of the Lucene RAG Library
 */
public class RAGExample {

    public static void main(String[] args) throws Exception {
        // Create index path (local file storage - no server needed!)
        Path indexPath = Paths.get("lucene-index");

        // Initialize embedding provider (use real embeddings in production)
        EmbeddingProvider embeddingProvider = new MockEmbeddingProvider(128);

        // Create RAG service - stores everything locally in files
        try (RAGService ragService = new RAGService(indexPath, embeddingProvider)) {

            // Add documents with metadata
            Map<String, String> metadata1 = new HashMap<>();
            metadata1.put("author", "Alice");
            metadata1.put("category", "AI");
            ragService.addDocument(
                "doc1",
                "Machine learning is a subset of artificial intelligence that focuses on data and algorithms",
                metadata1
            );

            Map<String, String> metadata2 = new HashMap<>();
            metadata2.put("author", "Bob");
            metadata2.put("category", "AI");
            ragService.addDocument(
                "doc2",
                "Deep learning uses neural networks with multiple layers to analyze data",
                metadata2
            );

            Map<String, String> metadata3 = new HashMap<>();
            metadata3.put("author", "Charlie");
            metadata3.put("category", "Programming");
            ragService.addDocument(
                "doc3",
                "Python is a high-level programming language known for its simplicity",
                metadata3
            );

            // Commit changes
            ragService.commit();

            System.out.println("Total documents indexed: " + ragService.getDocumentCount());
            System.out.println();

            // Perform semantic search
            System.out.println("=== Semantic Search Results ===");
            String query = "artificial intelligence and neural networks";
            List<SearchResult> results = ragService.search(query, 3);

            for (int i = 0; i < results.size(); i++) {
                SearchResult result = results.get(i);
                System.out.printf("%d. [Score: %.4f] %s%n",
                    i + 1, result.getScore(), result.getContent());
                System.out.printf("   Author: %s, Category: %s%n",
                    result.getMetadata("author"), result.getMetadata("category"));
                System.out.println();
            }

            // Retrieve context for RAG
            System.out.println("=== Retrieved Context for RAG ===");
            String context = ragService.retrieveContext(query, 2);
            System.out.println(context);
            System.out.println();

            // Hybrid search
            System.out.println("=== Hybrid Search Results ===");
            List<SearchResult> hybridResults = ragService.hybridSearch("learning", 2, 0.7f);
            for (SearchResult result : hybridResults) {
                System.out.printf("[Score: %.4f] %s%n", result.getScore(), result.getContent());
            }
        }

        System.out.println("\n✓ All data stored locally in: " + indexPath.toAbsolutePath());
        System.out.println("✓ No external server required!");
    }
}


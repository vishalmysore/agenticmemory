package io.github.vishalmysore.rag.examples;

import io.github.vishalmysore.*;

import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Example demonstrating how to load previously indexed documents from the Lucene index.
 * This shows that the RAG system persists data to disk and can reload it later.
 */
public class LoadPreviousDocumentsExample {

    public static void main(String[] args) throws Exception {
        System.out.println("=== Load Previous Documents Example ===\n");

        // Use a persistent index path
        var indexPath = Paths.get("persistent-rag-index");
        var embeddings = new MockEmbeddingProvider(128);

        // PART 1: Index some documents (first run)
        System.out.println("PART 1: Creating and indexing documents...\n");
        try (RAGService rag = new RAGService(indexPath, embeddings)) {
            
            // Add documents with metadata
            Map<String, String> metadata1 = new HashMap<>();
            metadata1.put("author", "Alice");
            metadata1.put("date", "2025-12-09");
            metadata1.put("category", "AI");
            
            rag.addDocument("doc1", 
                "Machine learning is a subset of artificial intelligence.", 
                metadata1);

            Map<String, String> metadata2 = new HashMap<>();
            metadata2.put("author", "Bob");
            metadata2.put("date", "2025-12-08");
            metadata2.put("category", "Programming");
            
            rag.addDocument("doc2", 
                "Python is a versatile programming language.", 
                metadata2);

            Map<String, String> metadata3 = new HashMap<>();
            metadata3.put("author", "Charlie");
            metadata3.put("date", "2025-12-07");
            metadata3.put("category", "AI");
            
            rag.addDocument("doc3", 
                "Neural networks are inspired by biological brains.", 
                metadata3);

            rag.commit();
            
            int count = rag.getDocumentCount();
            System.out.println("✓ Indexed " + count + " documents");
            System.out.println("✓ Index saved to: " + indexPath.toAbsolutePath());
        }

        System.out.println("\n" + "=".repeat(60) + "\n");

        // PART 2: Load the previously indexed documents (simulates new session)
        System.out.println("PART 2: Loading documents from existing index...\n");
        try (RAGService rag = new RAGService(indexPath, embeddings)) {
            
            // Check how many documents exist
            int totalDocs = rag.getDocumentCount();
            System.out.println("Found " + totalDocs + " documents in the index\n");

            // Retrieve a specific document by ID
            System.out.println("--- Retrieving Specific Document ---");
            Document doc = rag.getDocumentById("doc2");
            if (doc != null) {
                System.out.println("ID: " + doc.getId());
                System.out.println("Content: " + doc.getContent());
                System.out.println("Metadata:");
                doc.getMetadata().forEach((key, value) -> 
                    System.out.println("  " + key + ": " + value));
                System.out.println("Vector dimension: " + 
                    (doc.getVector() != null ? doc.getVector().length : "none"));
            }

            System.out.println("\n--- Retrieving All Documents ---");
            List<Document> allDocs = rag.getAllDocuments();
            for (int i = 0; i < allDocs.size(); i++) {
                Document d = allDocs.get(i);
                System.out.printf("\n%d. [ID: %s] %s%n", 
                    i + 1, d.getId(), d.getContent());
                System.out.println("   Author: " + d.getMetadata("author") + 
                    ", Category: " + d.getMetadata("category"));
            }

            // Check if a document exists
            System.out.println("\n--- Checking Document Existence ---");
            System.out.println("Does 'doc1' exist? " + rag.documentExists("doc1"));
            System.out.println("Does 'doc999' exist? " + rag.documentExists("doc999"));

            // Search the loaded documents
            System.out.println("\n--- Searching Loaded Documents ---");
            List<SearchResult> results = rag.search("artificial intelligence", 2);
            System.out.println("Query: 'artificial intelligence'");
            System.out.println("Top 2 results:");
            for (int i = 0; i < results.size(); i++) {
                SearchResult result = results.get(i);
                System.out.printf("\n%d. [Score: %.4f] %s%n", 
                    i + 1, result.getScore(), result.getContent());
                System.out.println("   By: " + result.getMetadata("author"));
            }

            // Add a new document to the existing index
            System.out.println("\n--- Adding New Document to Existing Index ---");
            Map<String, String> metadata4 = new HashMap<>();
            metadata4.put("author", "David");
            metadata4.put("date", "2025-12-09");
            metadata4.put("category", "Cloud");
            
            rag.addDocument("doc4", 
                "Cloud computing provides on-demand computing resources.", 
                metadata4);
            rag.commit();

            System.out.println("✓ Added new document");
            System.out.println("✓ Total documents now: " + rag.getDocumentCount());
        }

        System.out.println("\n=== Example Complete ===");
        System.out.println("✓ Documents persist across sessions");
        System.out.println("✓ Can retrieve individual or all documents");
        System.out.println("✓ Can check existence and search loaded data");
        System.out.println("✓ Can add to existing indexes");
    }
}

package io.github.vishalmysore.rag.examples;

import io.github.vishalmysore.Document;
import io.github.vishalmysore.OpenAIEmbeddingProvider;
import io.github.vishalmysore.RAGService;
import io.github.vishalmysore.SearchResult;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

/**
 * Example demonstrating RAG Q&A system using content from a text file.
 * This example reads multiple tech leaders' biographies from a text file, creates embeddings,
 * and allows interactive question-answering across all profiles.
 * 
 * Features:
 * - Multiple person biographies with structured sections
 * - Semantic search across all people to find relevant experts
 * - Questions like "Who has patents in ML?" or "Who worked at Google?"
 * - Demonstrates RAG's ability to find needles in haystacks
 * 
 * Usage:
 * 1. Set OPENAI_API_KEY environment variable
 * 2. Run the program
 * 3. Ask questions about any person, compare people, or search by expertise
 */
public class BioRAGExample {

    public static void main(String[] args) throws Exception {
        // Get API key from environment variable
        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null || apiKey.isEmpty()) {
            System.err.println("Error: OPENAI_API_KEY environment variable is required!");
            System.err.println("\nSet it with: $env:OPENAI_API_KEY=\"your-key-here\"");
            return;
        }

        System.out.println("=== Tech Leaders RAG Q&A System ===\n");
        System.out.println("Loading biographies from file...");

        // Read the tech leaders bio file
        String bioFilePath = "src/main/resources/tech_leaders.txt";
        String bioContent;
        try {
            bioContent = Files.readString(Paths.get(bioFilePath));
            System.out.println("✓ Loaded tech leaders file (" + bioContent.length() + " characters)\n");
        } catch (IOException e) {
            System.err.println("Error reading bio file: " + e.getMessage());
            System.err.println("Expected location: " + bioFilePath);
            return;
        }

        // Initialize OpenAI embedding provider
        System.out.println("Initializing OpenAI embeddings (text-embedding-3-small, 1024 dimensions)...");
        OpenAIEmbeddingProvider embeddings = new OpenAIEmbeddingProvider(
            apiKey,
            "text-embedding-3-small",
            1024
        );

        try {
            // Initialize RAG service
            try (RAGService rag = new RAGService(Paths.get("tech-leaders-index"), embeddings)) {

                // Split bios into person + section documents and index them
                System.out.println("Creating embeddings and indexing tech leaders...\n");
                indexTechLeaders(rag, bioContent);
                
                System.out.println("✓ Indexing complete! Total documents: " + rag.getDocumentCount());
                System.out.println();

                // Interactive Q&A mode
                System.out.println("=== Interactive Q&A Mode ===\n");
                System.out.println("Ask questions about any tech leader, compare them, or search by expertise.");
                System.out.println("\nSample Questions:");
                System.out.println("  - Who has patents in machine learning?");
                System.out.println("  - Which person worked at Google?");
                System.out.println("  - Who invented the World Wide Web?");
                System.out.println("  - Compare the educational backgrounds");
                System.out.println("  - Who has the most patents?");
                System.out.println("  - Tell me about women in computing");
                System.out.println("\nCommands:");
                System.out.println("  - Type your question to search");
                System.out.println("  - 'list' - Show all indexed people and sections");
                System.out.println("  - 'exit' or 'quit' - Exit the program\n");

                try (Scanner scanner = new Scanner(System.in)) {
                    while (true) {
                        System.out.print("Your Question: ");
                        String question = scanner.nextLine().trim();

                        if (question.equalsIgnoreCase("exit") || question.equalsIgnoreCase("quit")) {
                            System.out.println("\nGoodbye!");
                            break;
                        }

                        if (question.equalsIgnoreCase("list")) {
                            listAllSections(rag);
                            continue;
                        }

                        if (question.isEmpty()) {
                            continue;
                        }

                        // Answer the question
                        answerQuestion(rag, question);
                        System.out.println();
                    }
                }

                System.out.println("\n=== Session Complete ===");
                System.out.println("✓ All data saved to: tech-leaders-index/");
            }
        } finally {
            embeddings.close();
            System.out.println("✓ Resources cleaned up");
        }
    }

    /**
     * Index tech leaders by splitting content by person and then by sections
     */
    private static void indexTechLeaders(RAGService rag, String bioContent) throws Exception {
        // Split by person separator (===)
        String[] people = bioContent.split("(?m)^===\\s*$");
        
        int totalDocs = 0;
        for (String personBio : people) {
            personBio = personBio.trim();
            if (personBio.isEmpty() || personBio.length() < 100) {
                continue;
            }

            // Extract person name
            String personName = extractPersonName(personBio);
            System.out.println("  Indexing: " + personName);

            // Split person's bio into sections (by emoji markers)
            String[] sections = personBio.split("(?=👤|🎓|💼|🧑‍💻|🧪|📝|🔎)");
            
            int sectionCount = 0;
            for (String section : sections) {
                section = section.trim();
                if (section.isEmpty() || section.length() < 30) {
                    continue;
                }

                String docId = sanitizeId(personName) + "_section_" + (sectionCount + 1);
                String sectionTitle = extractSectionTitle(section);
                
                // Prepend person name to content for better context
                String contentWithPerson = "PERSON: " + personName + "\n\n" + section;
                
                System.out.println("    ✓ " + sectionTitle);
                rag.addDocument(docId, contentWithPerson);
                sectionCount++;
                totalDocs++;
            }
            
            System.out.println();
        }

        rag.commit();
        System.out.println("Total sections indexed: " + totalDocs);
        System.out.println();
    }

    /**
     * Extract person name from their bio
     */
    private static String extractPersonName(String personBio) {
        // Look for "PERSON: Name" pattern
        if (personBio.contains("PERSON:")) {
            String[] lines = personBio.split("\n");
            for (String line : lines) {
                if (line.contains("PERSON:")) {
                    return line.replace("👤", "").replace("PERSON:", "").trim();
                }
            }
        }
        
        // Fallback: use first line
        String firstLine = personBio.split("\n")[0].trim();
        return firstLine.substring(0, Math.min(30, firstLine.length()));
    }

    /**
     * Extract section title from content
     */
    private static String extractSectionTitle(String section) {
        String[] lines = section.split("\n");
        String firstLine = lines[0].trim();
        
        // Remove emoji and clean up
        firstLine = firstLine.replaceAll("[👤🎓💼🧑‍💻🧪📝🔎]", "").trim();
        
        // If line contains colon, take the part before it
        if (firstLine.contains(":")) {
            String title = firstLine.split(":")[0].trim();
            if (!title.isEmpty()) {
                return title;
            }
        }
        
        // Otherwise use first 40 characters
        return firstLine.substring(0, Math.min(40, firstLine.length()));
    }

    /**
     * Sanitize person name for use in document IDs
     */
    private static String sanitizeId(String name) {
        return name.toLowerCase()
                   .replaceAll("[^a-z0-9]+", "_")
                   .replaceAll("^_|_$", "");
    }

    /**
     * Answer a question using RAG
     */
    private static void answerQuestion(RAGService rag, String question) throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("Question: " + question);
        System.out.println("=".repeat(70));

        // Search for relevant content across all people
        List<SearchResult> results = rag.search(question, 5);

        if (results.isEmpty()) {
            System.out.println("\nNo relevant information found.");
            return;
        }

        // Display relevant sections
        System.out.println("\n📚 Relevant Information (Top " + results.size() + " matches):\n");

        for (int i = 0; i < results.size(); i++) {
            SearchResult result = results.get(i);
            
            // Extract person name from content
            String personName = extractPersonFromContent(result.getContent());
            
            System.out.printf("%d. %s [Relevance: %.1f%%]\n", 
                i + 1, personName, result.getScore() * 100);
            
            // Show first 250 characters of content
            String preview = result.getContent();
            if (preview.length() > 250) {
                preview = preview.substring(0, 250) + "...";
            }
            System.out.println("   " + preview.replace("\n", "\n   "));
            System.out.println();
        }

        // Get combined context
        String context = rag.retrieveContext(question, 3);
        
        System.out.println("💡 Answer Summary:");
        System.out.println("   Based on " + Math.min(3, results.size()) + " most relevant sections");
        System.out.println("   Context length: " + context.length() + " characters");
        System.out.println();
        System.out.println("   💬 This context can be sent to an LLM like GPT-4 to generate a detailed answer");
        System.out.println("      with citations and comparisons across multiple people.");
    }

    /**
     * Extract person name from content
     */
    private static String extractPersonFromContent(String content) {
        if (content.contains("PERSON:")) {
            String[] lines = content.split("\n");
            for (String line : lines) {
                if (line.contains("PERSON:")) {
                    return line.replace("👤", "").replace("PERSON:", "").trim();
                }
            }
        }
        return "Unknown";
    }

    /**
     * List all indexed sections
     */
    private static void listAllSections(RAGService rag) throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("Indexed Bio Sections");
        System.out.println("=".repeat(70) + "\n");

        List<Document> allDocs = rag.getAllDocuments();
        
        System.out.println("Total sections: " + allDocs.size() + "\n");

        for (int i = 0; i < allDocs.size(); i++) {
            Document doc = allDocs.get(i);
            String preview = doc.getContent();
            if (preview.length() > 100) {
                preview = preview.substring(0, 100) + "...";
            }
            System.out.printf("%d. [ID: %s]\n", i + 1, doc.getId());
            System.out.println("   " + preview.replace("\n", " "));
            System.out.println();
        }
    }
}

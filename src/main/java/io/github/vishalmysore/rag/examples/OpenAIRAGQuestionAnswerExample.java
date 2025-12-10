package io.github.vishalmysore.rag.examples;

import io.github.vishalmysore.*;

import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

/**
 * Complete example demonstrating OpenAI-based RAG question-answering system.
 * This example:
 * 1. Takes OpenAI configuration (base URL, API key, model)
 * 2. Creates embeddings for documents
 * 3. Allows asking questions and retrieves relevant context
 * 4. Can be used with any OpenAI-compatible API (OpenAI, Azure OpenAI, local models, etc.)
 */
public class OpenAIRAGQuestionAnswerExample {

    public static void main(String[] args) throws Exception {
        // Configuration - can be passed as arguments or environment variables
        String baseUrl = getConfigValue(args, 0, "OPENAI_BASE_URL", "https://api.openai.com/v1/embeddings");
        String apiKey = getConfigValue(args, 1, "OPENAI_API_KEY", null);
        String model = getConfigValue(args, 2, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small");
        int dimension = Integer.parseInt(getConfigValue(args, 3, "OPENAI_EMBEDDING_DIMENSION", "1536"));

        if (apiKey == null || apiKey.isEmpty()) {
            System.err.println("Error: OpenAI API key is required!");
            System.err.println("\nUsage options:");
            System.err.println("1. Set OPENAI_API_KEY environment variable");
            System.err.println("2. Pass as arguments: <baseUrl> <apiKey> <model> <dimension>");
            System.err.println("\nExample:");
            System.err.println("  java OpenAIRAGQuestionAnswerExample https://api.openai.com/v1/embeddings sk-xxx text-embedding-3-small 1536");
            return;
        }

        System.out.println("=== OpenAI RAG Question-Answer System ===\n");
        System.out.println("Configuration:");
        System.out.println("  Base URL: " + baseUrl);
        System.out.println("  Model: " + model);
        System.out.println("  Dimension: " + dimension);
        System.out.println("  API Key: " + maskApiKey(apiKey));
        System.out.println();

        // Initialize OpenAI embedding provider
        OpenAIEmbeddingProvider embeddings = new OpenAIEmbeddingProvider(baseUrl, apiKey, model, dimension);

        try {
            // Initialize RAG service
            try (RAGService rag = new RAGService(Paths.get("openai-qa-index"), embeddings)) {

                // Step 1: Index knowledge base documents
                System.out.println("Step 1: Creating knowledge base with embeddings...\n");
                indexKnowledgeBase(rag);
                
                System.out.println("✓ Successfully created embeddings for all documents");
                System.out.println("✓ Total documents indexed: " + rag.getDocumentCount());
                System.out.println();

                // Step 2: Interactive question-answering
                System.out.println("Step 2: Question-Answering Mode\n");
                System.out.println("You can now ask questions about the knowledge base.");
                System.out.println("Type 'list' to see all documents, 'exit' to quit.\n");

                try (Scanner scanner = new Scanner(System.in)) {
                    while (true) {
                        System.out.print("Your Question: ");
                        String question = scanner.nextLine().trim();

                        if (question.equalsIgnoreCase("exit") || question.equalsIgnoreCase("quit")) {
                            System.out.println("\nGoodbye!");
                            break;
                        }

                        if (question.equalsIgnoreCase("list")) {
                            listAllDocuments(rag);
                            continue;
                        }

                        if (question.isEmpty()) {
                            continue;
                        }

                        // Answer the question using RAG
                        answerQuestion(rag, question);
                        System.out.println();
                    }
                }

                System.out.println("\n=== Session Complete ===");
                System.out.println("✓ All data saved to: openai-qa-index/");
                System.out.println("✓ Restart the program to reload the knowledge base");
            }
        } finally {
            embeddings.close();
            System.out.println("✓ Resources cleaned up");
        }
    }

    /**
     * Index a knowledge base with various documents
     */
    private static void indexKnowledgeBase(RAGService rag) throws Exception {
        System.out.println("Indexing documents (creating embeddings)...\n");

        // Technology and AI documents
        Map<String, String> meta1 = new HashMap<>();
        meta1.put("category", "AI");
        meta1.put("author", "System");
        meta1.put("date", "2025-12-09");
        
        rag.addDocument("ai_ml", 
            "Artificial Intelligence (AI) is the simulation of human intelligence by machines. " +
            "Machine Learning is a subset of AI that enables systems to learn and improve from experience " +
            "without being explicitly programmed. Deep learning, a subset of machine learning, uses neural " +
            "networks with multiple layers to process data and extract patterns.",
            meta1);
        System.out.println("✓ Indexed: AI and Machine Learning");

        Map<String, String> meta2 = new HashMap<>();
        meta2.put("category", "Technology");
        meta2.put("author", "System");
        
        rag.addDocument("python_lang",
            "Python is a high-level, interpreted programming language known for its simplicity and readability. " +
            "It is widely used in data science, machine learning, web development, and automation. " +
            "Python has extensive libraries like NumPy, pandas, TensorFlow, and PyTorch that make it ideal for AI development.",
            meta2);
        System.out.println("✓ Indexed: Python Programming");

        Map<String, String> meta3 = new HashMap<>();
        meta3.put("category", "Technology");
        
        rag.addDocument("java_lang",
            "Java is a robust, object-oriented programming language designed for cross-platform compatibility. " +
            "It runs on the Java Virtual Machine (JVM) and follows the 'write once, run anywhere' principle. " +
            "Java is extensively used in enterprise applications, Android development, and large-scale systems.",
            meta3);
        System.out.println("✓ Indexed: Java Programming");

        Map<String, String> meta4 = new HashMap<>();
        meta4.put("category", "AI");
        
        rag.addDocument("nlp",
            "Natural Language Processing (NLP) is a branch of AI that focuses on the interaction between " +
            "computers and human language. NLP enables machines to understand, interpret, and generate human language. " +
            "Applications include chatbots, machine translation, sentiment analysis, and text summarization.",
            meta4);
        System.out.println("✓ Indexed: Natural Language Processing");

        Map<String, String> meta5 = new HashMap<>();
        meta5.put("category", "AI");
        
        rag.addDocument("neural_networks",
            "Neural networks are computing systems inspired by biological brains. They consist of interconnected " +
            "nodes (neurons) organized in layers. Deep neural networks with many layers are used for complex tasks " +
            "like image recognition, speech processing, and game playing. Training involves adjusting connection weights " +
            "through backpropagation.",
            meta5);
        System.out.println("✓ Indexed: Neural Networks");

        Map<String, String> meta6 = new HashMap<>();
        meta6.put("category", "Technology");
        
        rag.addDocument("cloud_computing",
            "Cloud computing provides on-demand access to computing resources over the internet. " +
            "Major providers include AWS, Azure, and Google Cloud. Cloud services offer scalability, " +
            "cost-effectiveness, and flexibility. Common models include IaaS, PaaS, and SaaS.",
            meta6);
        System.out.println("✓ Indexed: Cloud Computing");

        Map<String, String> meta7 = new HashMap<>();
        meta7.put("category", "Database");
        
        rag.addDocument("databases",
            "Databases are organized collections of structured data. SQL databases like PostgreSQL and MySQL " +
            "use relational models with tables and schemas. NoSQL databases like MongoDB and Cassandra offer " +
            "flexible schemas for unstructured data. Vector databases store and search high-dimensional vectors " +
            "for AI applications.",
            meta7);
        System.out.println("✓ Indexed: Databases");

        Map<String, String> meta8 = new HashMap<>();
        meta8.put("category", "AI");
        
        rag.addDocument("rag",
            "Retrieval-Augmented Generation (RAG) is an AI technique that combines information retrieval with " +
            "text generation. RAG systems first retrieve relevant documents from a knowledge base, then use that " +
            "context to generate accurate, grounded responses. This approach reduces hallucinations and enables " +
            "LLMs to access up-to-date information beyond their training data.",
            meta8);
        System.out.println("✓ Indexed: RAG (Retrieval-Augmented Generation)");

        // Commit all documents
        rag.commit();
        System.out.println();
    }

    /**
     * Answer a question using RAG approach
     */
    private static void answerQuestion(RAGService rag, String question) throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("Question: " + question);
        System.out.println("=".repeat(70));

        // Search for relevant documents
        List<SearchResult> results = rag.search(question, 3);

        if (results.isEmpty()) {
            System.out.println("\nNo relevant information found in the knowledge base.");
            return;
        }

        // Display relevant context
        System.out.println("\n📚 Relevant Information (Top " + results.size() + " matches):\n");

        for (int i = 0; i < results.size(); i++) {
            SearchResult result = results.get(i);
            System.out.printf("%d. [Relevance: %.2f%%] %s\n", 
                i + 1, result.getScore() * 100, result.getId());
            System.out.println("   Category: " + result.getMetadata("category"));
            System.out.println("   " + result.getContent());
            System.out.println();
        }

        // Get combined context for answer generation
        String context = rag.retrieveContext(question, 2);
        
        System.out.println("💡 Answer Context:");
        System.out.println("   (This context would be sent to an LLM like GPT-4 for answer generation)");
        System.out.println();
        System.out.println("   Context length: " + context.length() + " characters");
        System.out.println("   Based on: " + Math.min(2, results.size()) + " most relevant documents");
        System.out.println();
        System.out.println("📝 Suggested Prompt for LLM:");
        System.out.println("   \"Based on the following context, answer the question.");
        System.out.println("   Context: " + context.substring(0, Math.min(100, context.length())) + "...");
        System.out.println("   Question: " + question + "\"");
    }

    /**
     * List all documents in the knowledge base
     */
    private static void listAllDocuments(RAGService rag) throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("Knowledge Base Documents");
        System.out.println("=".repeat(70) + "\n");

        List<Document> allDocs = rag.getAllDocuments();
        
        System.out.println("Total documents: " + allDocs.size() + "\n");

        for (int i = 0; i < allDocs.size(); i++) {
            Document doc = allDocs.get(i);
            System.out.printf("%d. [ID: %s]\n", i + 1, doc.getId());
            System.out.println("   Category: " + doc.getMetadata("category"));
            System.out.println("   Content: " + doc.getContent().substring(0, Math.min(100, doc.getContent().length())) + "...");
            System.out.println();
        }
    }

    /**
     * Get configuration value from args, environment, or default
     */
    private static String getConfigValue(String[] args, int index, String envVar, String defaultValue) {
        if (args.length > index && args[index] != null && !args[index].isEmpty()) {
            return args[index];
        }
        String envValue = System.getenv(envVar);
        if (envValue != null && !envValue.isEmpty()) {
            return envValue;
        }
        return defaultValue;
    }

    /**
     * Mask API key for display
     */
    private static String maskApiKey(String apiKey) {
        if (apiKey == null || apiKey.length() < 8) {
            return "***";
        }
        return apiKey.substring(0, 7) + "..." + apiKey.substring(apiKey.length() - 4);
    }
}

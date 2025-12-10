package io.github.vishalmysore.rag.examples;

import io.github.vishalmysore.ChunkingStrategy;
import io.github.vishalmysore.OpenAIEmbeddingProvider;
import io.github.vishalmysore.RAGService;
import io.github.vishalmysore.SearchResult;
import io.github.vishalmysore.chunking.*;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * Comprehensive demonstration of 9 advanced chunking strategies for RAG systems.
 * 
 * This example shows how to use the extensible chunking framework where all strategies
 * are part of the core module (io.github.vishalmysore.chunking package).
 * 
 * Anyone can create custom chunking strategies by implementing the ChunkingStrategy interface.
 * 
 * Usage:
 * 1. Set OPENAI_API_KEY environment variable
 * 2. Run: mvn exec:java -q
 * 3. Select a chunking strategy to demo (1-9) or 0 to exit
 */
public class ChunkingStrategiesExample {

    private static final Scanner scanner = new Scanner(System.in);

    public static void main(String[] args) throws Exception {
        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null || apiKey.isEmpty()) {
            System.err.println("Error: OPENAI_API_KEY environment variable is required!");
            return;
        }

        System.out.println("=== Advanced Chunking Strategies for RAG Systems ===\n");

        OpenAIEmbeddingProvider embeddings = new OpenAIEmbeddingProvider(apiKey, "text-embedding-3-small", 1024);

        try {
            while (true) {
                displayMenu();
                int choice = getUserChoice();

                if (choice == 0) {
                    System.out.println("\nExiting...");
                    break;
                }

                demonstrateStrategy(choice, embeddings);
                
                System.out.print("\nPress Enter to continue...");
                scanner.nextLine();
            }
        } finally {
            embeddings.close();
            System.out.println("✓ Resources cleaned up");
        }
    }

    private static void displayMenu() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("SELECT A CHUNKING STRATEGY TO DEMONSTRATE:");
        System.out.println("=".repeat(70));
        System.out.println("1. Sliding Window Chunking (Healthcare Patient Notes)");
        System.out.println("2. Adaptive Chunking (Legal Contracts)");
        System.out.println("3. Entity-Based Chunking (News Articles)");
        System.out.println("4. Topic/Theme-Based Chunking (Tech Leaders - Semantic Grouping)");
        System.out.println("5. Hybrid Chunking (Software Documentation)");
        System.out.println("6. Task-Aware Chunking (Code Analysis - Different Tasks)");
        System.out.println("7. HTML/XML Tag-Based Splitting (Web Content)");
        System.out.println("8. Code-Specific Splitting (Python Code - AST-based)");
        System.out.println("9. Regular Expression Splitting (Server Logs)");
        System.out.println("0. Exit");
        System.out.println("=".repeat(70));
        System.out.print("\nEnter your choice (0-9): ");
    }

    private static int getUserChoice() {
        try {
            String input = scanner.nextLine().trim();
            return Integer.parseInt(input);
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    private static void demonstrateStrategy(int choice, OpenAIEmbeddingProvider embeddings) throws Exception {
        switch (choice) {
            case 1: demonstrateSlidingWindow(embeddings); break;
            case 2: demonstrateAdaptive(embeddings); break;
            case 3: demonstrateEntityBased(embeddings); break;
            case 4: demonstrateTopicBased(embeddings); break;
            case 5: demonstrateHybrid(embeddings); break;
            case 6: demonstrateTaskAware(embeddings); break;
            case 7: demonstrateHTMLTagBased(embeddings); break;
            case 8: demonstrateCodeSpecific(embeddings); break;
            case 9: demonstrateRegexSplitting(embeddings); break;
            default: System.out.println("\nInvalid choice. Please try again.");
        }
    }

    /**
     * Strategy 1: Sliding Window Chunking
     * Uses the core SlidingWindowChunking strategy
     */
    private static void demonstrateSlidingWindow(OpenAIEmbeddingProvider embeddings) throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("STRATEGY 1: SLIDING WINDOW CHUNKING");
        System.out.println("Domain: Healthcare Patient Notes");
        System.out.println("=".repeat(70));

        String content = Files.readString(Paths.get("src/main/resources/patient_notes.txt"));
        
        // Create the strategy from core framework
        ChunkingStrategy strategy = new SlidingWindowChunking(150, 30);
        
        System.out.println("\n" + strategy.getDescription());
        System.out.println("Total words: " + content.split("\\s+").length);

        try (RAGService rag = new RAGService(Paths.get("sliding-window-index"), embeddings)) {
            // Use RAGService's chunking support
            int chunkCount = rag.addDocumentWithChunking("patient_notes", content, strategy);
            rag.commit();

            System.out.println("\n✓ Indexed " + chunkCount + " chunks");
            
            demoQuery(rag, "What were the patient's vital signs?");
            demoQuery(rag, "What treatment protocol was used?");
        }
    }

    /**
     * Strategy 2: Adaptive Chunking
     * Uses the core AdaptiveChunking strategy
     */
    private static void demonstrateAdaptive(OpenAIEmbeddingProvider embeddings) throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("STRATEGY 2: ADAPTIVE CHUNKING");
        System.out.println("Domain: Legal Contracts");
        System.out.println("=".repeat(70));

        String content = Files.readString(Paths.get("src/main/resources/legal_contract.txt"));

        // Create the strategy from core framework
        ChunkingStrategy strategy = new AdaptiveChunking("(?m)^SECTION \\d+:", 800, 1200);
        
        System.out.println("\n" + strategy.getDescription());

        try (RAGService rag = new RAGService(Paths.get("adaptive-index"), embeddings)) {
            int chunkCount = rag.addDocumentWithChunking("legal_contract", content, strategy);
            rag.commit();

            System.out.println("\n✓ Created " + chunkCount + " adaptive chunks - no mid-clause splits!");
            
            demoQuery(rag, "What are the payment terms?");
            demoQuery(rag, "Explain the confidentiality obligations");
        }
    }

    /**
     * Strategy 3: Entity-Based Chunking
     * Uses the core EntityBasedChunking strategy
     */
    private static void demonstrateEntityBased(OpenAIEmbeddingProvider embeddings) throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("STRATEGY 3: ENTITY-BASED CHUNKING");
        System.out.println("Domain: News Articles");
        System.out.println("=".repeat(70));

        String content = Files.readString(Paths.get("src/main/resources/news_articles.txt"));

        // Create the strategy from core framework
        String[] entities = {"Elon Musk", "Tesla", "SpaceX", "Mark Zuckerberg", "Jeff Bezos", "Tim Cook"};
        ChunkingStrategy strategy = new EntityBasedChunking(entities);
        
        System.out.println("\n" + strategy.getDescription());

        try (RAGService rag = new RAGService(Paths.get("entity-based-index"), embeddings)) {
            int chunkCount = rag.addDocumentWithChunking("news_articles", content, strategy);
            rag.commit();

            System.out.println("\n✓ Created " + chunkCount + " entity-centric chunks!");
            
            demoQuery(rag, "What did Elon Musk announce?");
            demoQuery(rag, "Tell me about SpaceX activities");
        }
    }

    /**
     * Strategy 4: Topic/Theme-Based Chunking
     * Uses the core TopicBasedChunking strategy
     */
    private static void demonstrateTopicBased(OpenAIEmbeddingProvider embeddings) throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("STRATEGY 4: TOPIC/THEME-BASED CHUNKING");
        System.out.println("Domain: Tech Leaders (Semantic Grouping)");
        System.out.println("=".repeat(70));

        String content = Files.readString(Paths.get("src/main/resources/tech_leaders.txt"));

        // Create the strategy from core framework
        ChunkingStrategy strategy = new TopicBasedChunking("(🎓 EDUCATION:|💼 CAREER:|🧑‍💻 PATENTS:|📝 PUBLICATIONS:)");
        
        System.out.println("\n" + strategy.getDescription());

        try (RAGService rag = new RAGService(Paths.get("topic-based-index"), embeddings)) {
            int chunkCount = rag.addDocumentWithChunking("tech_leaders", content, strategy);
            rag.commit();

            System.out.println("\n✓ Created " + chunkCount + " topic-based chunks!");
            
            demoQuery(rag, "Compare educational backgrounds");
            demoQuery(rag, "Who has patents?");
        }
    }

    /**
     * Strategy 5: Hybrid Chunking
     * Uses the core HybridChunking strategy combining multiple strategies
     */
    private static void demonstrateHybrid(OpenAIEmbeddingProvider embeddings) throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("STRATEGY 5: HYBRID CHUNKING");
        System.out.println("Domain: Software Documentation");
        System.out.println("=".repeat(70));

        String content = Files.readString(Paths.get("src/main/resources/tech_leaders.txt"));

        // Create a hybrid strategy combining multiple approaches
        ChunkingStrategy adaptiveStrategy = new AdaptiveChunking("(?m)^===\\s*$");
        ChunkingStrategy topicStrategy = new TopicBasedChunking("(🎓 EDUCATION:|💼 CAREER:|🧑‍💻 PATENTS:|📝 PUBLICATIONS:)");
        
        // Combine them in a pipeline
        ChunkingStrategy strategy = new HybridChunking(adaptiveStrategy, topicStrategy);
        
        System.out.println("\n" + strategy.getDescription());

        try (RAGService rag = new RAGService(Paths.get("hybrid-index"), embeddings)) {
            int chunkCount = rag.addDocumentWithChunking("tech_docs", content, strategy);
            rag.commit();

            System.out.println("\n✓ Created " + chunkCount + " chunks through hybrid pipeline!");
            
            demoQuery(rag, "Who worked on compilers?");
            demoQuery(rag, "Tell me about deep learning pioneers");
        }
    }

    /**
     * Strategy 6: Task-Aware Chunking
     * Uses the core TaskAwareChunking strategy
     */
    private static void demonstrateTaskAware(OpenAIEmbeddingProvider embeddings) throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("STRATEGY 6: TASK-AWARE CHUNKING");
        System.out.println("Domain: Code Analysis (Python code)");
        System.out.println("=".repeat(70));

        String code = Files.readString(Paths.get("src/main/resources/sample_code.py"));

        System.out.println("\nDemonstrating 3 different task-specific chunking approaches:\n");

        // Task 1: Summarization
        demonstrateTask("SUMMARIZATION", code, embeddings, TaskAwareChunking.TaskType.SUMMARIZATION);
        
        // Task 2: Search
        demonstrateTask("SEARCH/RETRIEVAL", code, embeddings, TaskAwareChunking.TaskType.SEARCH);
        
        // Task 3: Q&A
        demonstrateTask("Q&A", code, embeddings, TaskAwareChunking.TaskType.QA);

        System.out.println("\n✓ Same code, 3 different chunking strategies for 3 tasks!");
    }

    private static void demonstrateTask(String taskName, String code, OpenAIEmbeddingProvider embeddings, 
                                       TaskAwareChunking.TaskType taskType) throws Exception {
        System.out.println("TASK: " + taskName);
        
        ChunkingStrategy strategy = new TaskAwareChunking(taskType);
        System.out.println("  " + strategy.getDescription());
        
        String indexPath = "task-" + taskType.name().toLowerCase() + "-index";
        try (RAGService rag = new RAGService(Paths.get(indexPath), embeddings)) {
            int chunkCount = rag.addDocumentWithChunking("python_code", code, strategy);
            rag.commit();
            System.out.println("  ✓ " + chunkCount + " chunks indexed\n");
        }
    }

    /**
     * Strategy 7: HTML/XML Tag-Based Chunking
     * Uses the core HTMLTagBasedChunking strategy
     */
    private static void demonstrateHTMLTagBased(OpenAIEmbeddingProvider embeddings) throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("STRATEGY 7: HTML/XML TAG-BASED SPLITTING");
        System.out.println("Domain: Web Content");
        System.out.println("=".repeat(70));

        String htmlContent = """
            <article>
              <h1>Complete Guide to RAG Systems</h1>
              <h2>Introduction</h2>
              <p>Retrieval Augmented Generation combines information retrieval with language models...</p>
              <h2>Chunking Strategies</h2>
              <p>Effective chunking is crucial for RAG performance...</p>
              <h3>Sliding Window</h3>
              <p>Sliding window creates overlapping chunks...</p>
              <h3>Adaptive Chunking</h3>
              <p>Adaptive chunking respects natural boundaries...</p>
              <h2>Implementation</h2>
              <p>When implementing RAG systems, consider...</p>
            </article>
            """;

        // Create the strategy from core framework
        ChunkingStrategy strategy = new HTMLTagBasedChunking("h2", true);
        
        System.out.println("\n" + strategy.getDescription());

        try (RAGService rag = new RAGService(Paths.get("html-tag-index"), embeddings)) {
            int chunkCount = rag.addDocumentWithChunking("html_doc", htmlContent, strategy);
            rag.commit();

            System.out.println("\n✓ Created " + chunkCount + " chunks respecting HTML structure!");
        }
    }

    /**
     * Strategy 8: Code-Specific Chunking
     * Uses the core CodeSpecificChunking strategy
     */
    private static void demonstrateCodeSpecific(OpenAIEmbeddingProvider embeddings) throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("STRATEGY 8: CODE-SPECIFIC SPLITTING");
        System.out.println("Domain: Python Code (AST-inspired boundaries)");
        System.out.println("=".repeat(70));

        String code = Files.readString(Paths.get("src/main/resources/sample_code.py"));

        // Create the strategy from core framework
        ChunkingStrategy strategy = new CodeSpecificChunking(CodeSpecificChunking.Language.PYTHON);
        
        System.out.println("\n" + strategy.getDescription());

        try (RAGService rag = new RAGService(Paths.get("code-specific-index"), embeddings)) {
            int chunkCount = rag.addDocumentWithChunking("python_code", code, strategy);
            rag.commit();

            System.out.println("\n✓ Indexed " + chunkCount + " code units (classes + functions)!");
        }
    }

    /**
     * Strategy 9: Regular Expression Chunking
     * Uses the core RegexChunking strategy
     */
    private static void demonstrateRegexSplitting(OpenAIEmbeddingProvider embeddings) throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("STRATEGY 9: REGULAR EXPRESSION (REGEX) SPLITTING");
        System.out.println("Domain: Server Logs");
        System.out.println("=".repeat(70));

        String logs = Files.readString(Paths.get("src/main/resources/server_logs.txt"));

        // Create the strategy from core framework - split by timestamp, group by severity
        ChunkingStrategy strategy = new RegexChunking(
            "\\[\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\]",
            "(ERROR|WARN|INFO|DEBUG)"
        );
        
        System.out.println("\n" + strategy.getDescription());

        try (RAGService rag = new RAGService(Paths.get("regex-logs-index"), embeddings)) {
            int chunkCount = rag.addDocumentWithChunking("server_logs", logs, strategy);
            rag.commit();

            System.out.println("\n✓ Created " + chunkCount + " log groups by severity!");
            
            demoQuery(rag, "What database errors occurred?");
            demoQuery(rag, "Show me payment processing issues");
        }
    }

    private static void demoQuery(RAGService rag, String question) throws Exception {
        System.out.println("\n  Demo Query: \"" + question + "\"");
        List<SearchResult> results = rag.search(question, 2);
        
        if (results.isEmpty()) {
            System.out.println("    No results found");
            return;
        }

        SearchResult top = results.get(0);
        String preview = top.getContent().substring(0, Math.min(100, top.getContent().length()));
        System.out.printf("    Top Result (%.1f%% match): %s...%n", 
                         top.getScore() * 100, preview);
    }
}

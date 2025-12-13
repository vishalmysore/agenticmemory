package io.github.vishalmysore.rag;

/**
 * Simple implementation of ContextGenerator that creates basic document context.
 * For production use, replace with an LLM-based implementation.
 */
public class SimpleContextGenerator implements ContextGenerator {
    
    private final int summaryLength;
    
    /**
     * Creates a SimpleContextGenerator with default summary length.
     */
    public SimpleContextGenerator() {
        this(150);
    }
    
    /**
     * Creates a SimpleContextGenerator with custom summary length.
     * 
     * @param summaryLength Maximum characters for document summary
     */
    public SimpleContextGenerator(int summaryLength) {
        this.summaryLength = summaryLength;
    }
    
    @Override
    public String generateContext(String fullDocument, String chunk, int chunkIndex, int totalChunks) {
        // Generate a simple context header
        StringBuilder context = new StringBuilder();
        
        // Add document summary
        String docSummary = generateDocumentSummary(fullDocument);
        context.append("DOCUMENT CONTEXT: ").append(docSummary).append("\n");
        
        // Add position information
        context.append("CHUNK POSITION: Part ").append(chunkIndex + 1)
               .append(" of ").append(totalChunks);
        
        return context.toString();
    }
    
    @Override
    public String generateDocumentSummary(String fullDocument) {
        if (fullDocument == null || fullDocument.isEmpty()) {
            return "Empty document";
        }
        
        // Extract first paragraph or first N characters
        String summary = fullDocument.trim();
        
        // Try to get first paragraph
        int firstParagraph = summary.indexOf("\n\n");
        if (firstParagraph > 0 && firstParagraph < summaryLength) {
            summary = summary.substring(0, firstParagraph);
        } else {
            // Otherwise, take first N characters and try to break at sentence
            if (summary.length() > summaryLength) {
                summary = summary.substring(0, summaryLength);
                int lastPeriod = summary.lastIndexOf('.');
                if (lastPeriod > summaryLength / 2) {
                    summary = summary.substring(0, lastPeriod + 1);
                } else {
                    summary += "...";
                }
            }
        }
        
        return summary;
    }
}

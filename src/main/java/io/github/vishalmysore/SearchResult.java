package io.github.vishalmysore;

import java.util.Map;

/**
 * Represents a search result with the document, score, and metadata.
 */
public class SearchResult {
    private final String id;
    private final String content;
    private final float score;
    private final Map<String, String> metadata;

    public SearchResult(String id, String content, float score, Map<String, String> metadata) {
        this.id = id;
        this.content = content;
        this.score = score;
        this.metadata = metadata;
    }

    public String getId() {
        return id;
    }

    public String getContent() {
        return content;
    }

    public float getScore() {
        return score;
    }

    public Map<String, String> getMetadata() {
        return metadata;
    }

    public String getMetadata(String key) {
        return metadata != null ? metadata.get(key) : null;
    }

    @Override
    public String toString() {
        return "SearchResult{" +
                "id='" + id + '\'' +
                ", score=" + score +
                ", content='" + content + '\'' +
                ", metadata=" + metadata +
                '}';
    }
}


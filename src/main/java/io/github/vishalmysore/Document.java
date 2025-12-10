package io.github.vishalmysore;

import java.util.HashMap;
import java.util.Map;

/**
 * Represents a document with text content, vector embeddings, and metadata.
 */
public class Document {
    private final String id;
    private final String content;
    private final float[] vector;
    private final Map<String, String> metadata;

    private Document(Builder builder) {
        this.id = builder.id;
        this.content = builder.content;
        this.vector = builder.vector;
        this.metadata = builder.metadata;
    }

    public String getId() {
        return id;
    }

    public String getContent() {
        return content;
    }

    public float[] getVector() {
        return vector;
    }

    public Map<String, String> getMetadata() {
        return metadata;
    }

    public String getMetadata(String key) {
        return metadata.get(key);
    }

    public static class Builder {
        private String id;
        private String content;
        private float[] vector;
        private Map<String, String> metadata = new HashMap<>();

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Builder content(String content) {
            this.content = content;
            return this;
        }

        public Builder vector(float[] vector) {
            this.vector = vector;
            return this;
        }

        public Builder metadata(Map<String, String> metadata) {
            this.metadata = new HashMap<>(metadata);
            return this;
        }

        public Builder addMetadata(String key, String value) {
            this.metadata.put(key, value);
            return this;
        }

        public Document build() {
            if (id == null || id.isEmpty()) {
                throw new IllegalArgumentException("Document ID cannot be null or empty");
            }
            if (content == null) {
                throw new IllegalArgumentException("Document content cannot be null");
            }
            return new Document(this);
        }
    }
}



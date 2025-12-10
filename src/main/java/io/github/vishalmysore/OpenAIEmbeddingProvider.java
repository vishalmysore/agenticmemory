package io.github.vishalmysore;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import okhttp3.*;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * OpenAI embedding provider that uses the OpenAI API to generate embeddings.
 * Supports custom API URLs and various embedding models.
 */
public class OpenAIEmbeddingProvider implements EmbeddingProvider {

    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    
    private final String apiUrl;
    private final String apiKey;
    private final String model;
    private final int dimension;
    private final OkHttpClient client;
    private final Gson gson;

    /**
     * Creates an OpenAI embedding provider with custom configuration.
     *
     * @param apiUrl The OpenAI API endpoint URL (e.g., "https://api.openai.com/v1/embeddings")
     * @param apiKey The API key for authentication
     * @param model The embedding model to use (e.g., "text-embedding-3-small", "text-embedding-ada-002")
     * @param dimension The dimension of the embeddings (e.g., 1536 for ada-002, configurable for v3 models)
     */
    public OpenAIEmbeddingProvider(String apiUrl, String apiKey, String model, int dimension) {
        if (apiUrl == null || apiUrl.trim().isEmpty()) {
            throw new IllegalArgumentException("API URL cannot be null or empty");
        }
        if (apiKey == null || apiKey.trim().isEmpty()) {
            throw new IllegalArgumentException("API key cannot be null or empty");
        }
        if (model == null || model.trim().isEmpty()) {
            throw new IllegalArgumentException("Model cannot be null or empty");
        }
        if (dimension <= 0) {
            throw new IllegalArgumentException("Dimension must be positive");
        }

        this.apiUrl = apiUrl;
        this.apiKey = apiKey;
        this.model = model;
        this.dimension = dimension;
        this.gson = new Gson();
        this.client = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(30, TimeUnit.SECONDS)
                .writeTimeout(30, TimeUnit.SECONDS)
                .build();
    }

    /**
     * Creates an OpenAI embedding provider with default OpenAI endpoint.
     *
     * @param apiKey The API key for authentication
     * @param model The embedding model to use
     * @param dimension The dimension of the embeddings
     */
    public OpenAIEmbeddingProvider(String apiKey, String model, int dimension) {
        this("https://api.openai.com/v1/embeddings", apiKey, model, dimension);
    }

    @Override
    public float[] embed(String text) {
        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException("Text cannot be null or empty");
        }

        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("input", text);
        requestBody.addProperty("model", model);
        
        // Add dimension parameter for models that support it (e.g., text-embedding-3-small/large)
        if (model.startsWith("text-embedding-3-")) {
            requestBody.addProperty("dimensions", dimension);
        }

        String jsonBody = gson.toJson(requestBody);

        Request request = new Request.Builder()
                .url(apiUrl)
                .addHeader("Authorization", "Bearer " + apiKey)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(jsonBody, JSON))
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String errorBody = response.body() != null ? response.body().string() : "No error details";
                throw new RuntimeException("OpenAI API request failed with code " + response.code() + ": " + errorBody);
            }

            String responseBody = response.body().string();
            JsonObject jsonResponse = gson.fromJson(responseBody, JsonObject.class);
            
            JsonArray data = jsonResponse.getAsJsonArray("data");
            if (data == null || data.isEmpty()) {
                throw new RuntimeException("No embedding data in OpenAI response");
            }

            JsonArray embeddingArray = data.get(0).getAsJsonObject().getAsJsonArray("embedding");
            float[] embedding = new float[embeddingArray.size()];
            
            for (int i = 0; i < embeddingArray.size(); i++) {
                embedding[i] = embeddingArray.get(i).getAsFloat();
            }

            if (embedding.length != dimension) {
                throw new RuntimeException("Expected embedding dimension " + dimension + " but got " + embedding.length);
            }

            return embedding;

        } catch (IOException e) {
            throw new RuntimeException("Failed to generate embedding: " + e.getMessage(), e);
        }
    }

    @Override
    public int getDimension() {
        return dimension;
    }

    /**
     * Gets the API URL being used.
     * @return The API URL
     */
    public String getApiUrl() {
        return apiUrl;
    }

    /**
     * Gets the model being used.
     * @return The model name
     */
    public String getModel() {
        return model;
    }

    /**
     * Closes the HTTP client and releases resources.
     */
    public void close() {
        client.dispatcher().executorService().shutdown();
        client.connectionPool().evictAll();
    }
}

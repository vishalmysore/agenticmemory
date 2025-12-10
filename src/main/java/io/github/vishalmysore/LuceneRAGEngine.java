package io.github.vishalmysore;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Main RAG engine using Apache Lucene with KnnVectorField for similarity search.
 * This engine supports both traditional keyword search and vector-based semantic search.
 */
public class LuceneRAGEngine implements Closeable {
    private static final Logger logger = LoggerFactory.getLogger(LuceneRAGEngine.class);

    private static final String FIELD_ID = "id";
    private static final String FIELD_CONTENT = "content";
    private static final String FIELD_VECTOR = "vector";
    private static final String FIELD_METADATA_PREFIX = "meta_";

    private final Directory directory;
    private final Analyzer analyzer;
    private IndexWriter indexWriter;
    private DirectoryReader indexReader;
    private IndexSearcher indexSearcher;
    private final int vectorDimension;

    /**
     * Creates a new LuceneRAGEngine with the specified index directory and vector dimension.
     *
     * @param indexPath       Path to the Lucene index directory (local filesystem)
     * @param vectorDimension Dimension of the vector embeddings
     * @throws IOException if an I/O error occurs
     */
    public LuceneRAGEngine(Path indexPath, int vectorDimension) throws IOException {
        this(FSDirectory.open(indexPath), new StandardAnalyzer(), vectorDimension);
    }

    /**
     * Creates a new LuceneRAGEngine with custom directory and analyzer.
     *
     * @param directory       Lucene Directory for storing the index
     * @param analyzer        Analyzer for text processing
     * @param vectorDimension Dimension of the vector embeddings
     * @throws IOException if an I/O error occurs
     */
    public LuceneRAGEngine(Directory directory, Analyzer analyzer, int vectorDimension) throws IOException {
        this.directory = directory;
        this.analyzer = analyzer;
        this.vectorDimension = vectorDimension;
        initializeIndexWriter();
        logger.info("LuceneRAGEngine initialized with vector dimension: {}", vectorDimension);
    }

    private void initializeIndexWriter() throws IOException {
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND);
        this.indexWriter = new IndexWriter(directory, config);
    }

    /**
     * Indexes a document with its vector embedding.
     *
     * @param document The document to index
     * @throws IOException if an I/O error occurs
     */
    public void indexDocument(Document document) throws IOException {
        org.apache.lucene.document.Document luceneDoc = new org.apache.lucene.document.Document();

        // Add ID field
        luceneDoc.add(new StringField(FIELD_ID, document.getId(), Field.Store.YES));

        // Add content field (both stored and indexed)
        luceneDoc.add(new TextField(FIELD_CONTENT, document.getContent(), Field.Store.YES));

        // Add vector field if present
        if (document.getVector() != null) {
            if (document.getVector().length != vectorDimension) {
                throw new IllegalArgumentException(
                        String.format("Vector dimension mismatch. Expected %d, got %d",
                                vectorDimension, document.getVector().length));
            }
            luceneDoc.add(new KnnFloatVectorField(FIELD_VECTOR, document.getVector()));
            // Also store the vector for retrieval
            luceneDoc.add(new StoredField(FIELD_VECTOR + "_stored", vectorToString(document.getVector())));
        }

        // Add metadata fields
        if (document.getMetadata() != null) {
            for (Map.Entry<String, String> entry : document.getMetadata().entrySet()) {
                luceneDoc.add(new StringField(FIELD_METADATA_PREFIX + entry.getKey(),
                        entry.getValue(), Field.Store.YES));
            }
        }

        // Delete any existing document with the same ID and add the new one
        indexWriter.updateDocument(new Term(FIELD_ID, document.getId()), luceneDoc);
        logger.debug("Indexed document with ID: {}", document.getId());
    }

    /**
     * Indexes multiple documents in a batch.
     *
     * @param documents List of documents to index
     * @throws IOException if an I/O error occurs
     */
    public void indexDocuments(List<Document> documents) throws IOException {
        for (Document doc : documents) {
            indexDocument(doc);
        }
        commit();
        logger.info("Indexed {} documents", documents.size());
    }

    /**
     * Performs vector similarity search using KNN.
     *
     * @param queryVector The query vector
     * @param topK        Number of top results to return
     * @return List of search results
     * @throws IOException if an I/O error occurs
     */
    public List<SearchResult> vectorSearch(float[] queryVector, int topK) throws IOException {
        if (queryVector.length != vectorDimension) {
            throw new IllegalArgumentException(
                    String.format("Query vector dimension mismatch. Expected %d, got %d",
                            vectorDimension, queryVector.length));
        }

        refreshSearcher();

        KnnFloatVectorQuery knnQuery = new KnnFloatVectorQuery(FIELD_VECTOR, queryVector, topK);
        TopDocs topDocs = indexSearcher.search(knnQuery, topK);

        return buildSearchResults(topDocs);
    }

    /**
     * Performs traditional keyword-based search.
     *
     * @param queryText The query text
     * @param topK      Number of top results to return
     * @return List of search results
     * @throws IOException if an I/O error occurs
     */
    public List<SearchResult> keywordSearch(String queryText, int topK) throws IOException {
        refreshSearcher();

        Query query = new TermQuery(new Term(FIELD_CONTENT, queryText.toLowerCase()));
        TopDocs topDocs = indexSearcher.search(query, topK);

        return buildSearchResults(topDocs);
    }

    /**
     * Performs hybrid search combining vector similarity and keyword matching.
     *
     * @param queryVector      The query vector
     * @param queryText        The query text
     * @param topK             Number of top results to return
     * @param vectorWeight     Weight for vector search (0.0 to 1.0)
     * @return List of search results
     * @throws IOException if an I/O error occurs
     */
    public List<SearchResult> hybridSearch(float[] queryVector, String queryText,
                                           int topK, float vectorWeight) throws IOException {
        refreshSearcher();

        // Create vector query
        KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery(FIELD_VECTOR, queryVector, topK * 2);

        // Create keyword query
        Query keywordQuery = new TermQuery(new Term(FIELD_CONTENT, queryText.toLowerCase()));

        // Combine queries with weights
        BooleanQuery.Builder builder = new BooleanQuery.Builder();
        builder.add(new BoostQuery(vectorQuery, vectorWeight), BooleanClause.Occur.SHOULD);
        builder.add(new BoostQuery(keywordQuery, 1.0f - vectorWeight), BooleanClause.Occur.SHOULD);

        TopDocs topDocs = indexSearcher.search(builder.build(), topK);
        return buildSearchResults(topDocs);
    }

    /**
     * Retrieves a document by its ID.
     *
     * @param documentId The ID of the document to retrieve
     * @return The document, or null if not found
     * @throws IOException if an I/O error occurs
     */
    public Document getDocumentById(String documentId) throws IOException {
        refreshSearcher();

        Query query = new TermQuery(new Term(FIELD_ID, documentId));
        TopDocs topDocs = indexSearcher.search(query, 1);

        if (topDocs.scoreDocs.length == 0) {
            return null;
        }

        org.apache.lucene.document.Document luceneDoc = indexSearcher.storedFields().document(topDocs.scoreDocs[0].doc);
        return buildDocumentFromLucene(luceneDoc);
    }

    /**
     * Retrieves all documents from the index.
     *
     * @return List of all documents
     * @throws IOException if an I/O error occurs
     */
    public List<Document> getAllDocuments() throws IOException {
        refreshSearcher();
        List<Document> documents = new ArrayList<>();

        StoredFields storedFields = indexSearcher.storedFields();
        for (int i = 0; i < indexReader.maxDoc(); i++) {
            org.apache.lucene.document.Document luceneDoc = storedFields.document(i);
            // Check if document is not deleted by verifying it has an ID
            if (luceneDoc.get(FIELD_ID) != null) {
                documents.add(buildDocumentFromLucene(luceneDoc));
            }
        }

        logger.info("Retrieved {} documents from index", documents.size());
        return documents;
    }

    /**
     * Checks if a document with the given ID exists in the index.
     *
     * @param documentId The ID to check
     * @return true if the document exists, false otherwise
     * @throws IOException if an I/O error occurs
     */
    public boolean documentExists(String documentId) throws IOException {
        return getDocumentById(documentId) != null;
    }

    /**
     * Deletes a document by ID.
     *
     * @param documentId The ID of the document to delete
     * @throws IOException if an I/O error occurs
     */
    public void deleteDocument(String documentId) throws IOException {
        indexWriter.deleteDocuments(new Term(FIELD_ID, documentId));
        logger.debug("Deleted document with ID: {}", documentId);
    }

    /**
     * Commits all pending changes to the index.
     *
     * @throws IOException if an I/O error occurs
     */
    public void commit() throws IOException {
        indexWriter.commit();
        logger.debug("Index committed");
    }

    /**
     * Gets the total number of documents in the index.
     *
     * @return The number of documents
     * @throws IOException if an I/O error occurs
     */
    public int getDocumentCount() throws IOException {
        refreshSearcher();
        return indexReader.numDocs();
    }

    private void refreshSearcher() throws IOException {
        if (indexReader == null) {
            indexReader = DirectoryReader.open(indexWriter);
            indexSearcher = new IndexSearcher(indexReader);
        } else {
            DirectoryReader newReader = DirectoryReader.openIfChanged(indexReader);
            if (newReader != null) {
                indexReader.close();
                indexReader = newReader;
                indexSearcher = new IndexSearcher(indexReader);
            }
        }
    }

    private List<SearchResult> buildSearchResults(TopDocs topDocs) throws IOException {
        List<SearchResult> results = new ArrayList<>();
        StoredFields storedFields = indexSearcher.storedFields();

        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            org.apache.lucene.document.Document doc = storedFields.document(scoreDoc.doc);

            String id = doc.get(FIELD_ID);
            String content = doc.get(FIELD_CONTENT);
            float score = scoreDoc.score;

            // Extract metadata
            Map<String, String> metadata = new HashMap<>();
            for (IndexableField field : doc.getFields()) {
                String fieldName = field.name();
                if (fieldName.startsWith(FIELD_METADATA_PREFIX)) {
                    String metaKey = fieldName.substring(FIELD_METADATA_PREFIX.length());
                    metadata.put(metaKey, field.stringValue());
                }
            }

            results.add(new SearchResult(id, content, score, metadata));
        }

        return results;
    }

    private String vectorToString(float[] vector) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < vector.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(vector[i]);
        }
        return sb.toString();
    }

    private float[] stringToVector(String vectorString) {
        if (vectorString == null || vectorString.isEmpty()) {
            return null;
        }
        String[] parts = vectorString.split(",");
        float[] vector = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            vector[i] = Float.parseFloat(parts[i].trim());
        }
        return vector;
    }

    private Document buildDocumentFromLucene(org.apache.lucene.document.Document luceneDoc) {
        String id = luceneDoc.get(FIELD_ID);
        String content = luceneDoc.get(FIELD_CONTENT);
        String vectorString = luceneDoc.get(FIELD_VECTOR + "_stored");
        float[] vector = stringToVector(vectorString);

        // Extract metadata
        Map<String, String> metadata = new HashMap<>();
        for (IndexableField field : luceneDoc.getFields()) {
            String fieldName = field.name();
            if (fieldName.startsWith(FIELD_METADATA_PREFIX)) {
                String metaKey = fieldName.substring(FIELD_METADATA_PREFIX.length());
                metadata.put(metaKey, field.stringValue());
            }
        }

        Document.Builder builder = new Document.Builder()
                .id(id)
                .content(content);

        if (vector != null) {
            builder.vector(vector);
        }

        if (!metadata.isEmpty()) {
            builder.metadata(metadata);
        }

        return builder.build();
    }

    @Override
    public void close() throws IOException {
        if (indexReader != null) {
            indexReader.close();
        }
        if (indexWriter != null) {
            indexWriter.close();
        }
        if (directory != null) {
            directory.close();
        }
        logger.info("LuceneRAGEngine closed");
    }
}


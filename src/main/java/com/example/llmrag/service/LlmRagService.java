package com.example.llmrag.service;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;
import jakarta.annotation.PostConstruct;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Service;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Service
public class LlmRagService {

    private static final String OPENAI_API_URL = "https://api.openai.com/v1";
    //private static final String GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent";
    private static final String GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent";
    private static final String CLAUDE_API_URL = "https://api.anthropic.com/v1/messages";

    // Replace with your API keys
    @Value("${openai.api.key}")
    String OPENAI_API_KEY;
    @Value("${google.api.key}")
    String GEMINI_API_KEY;
    @Value("${anthropic.api.key}")
    String CLAUDE_API_KEY;

    @Value("${llm.openai.enabled:false}")
    private boolean openAiEnabled;
    @Value("${llm.gemini.enabled:false}")
    private boolean geminiEnabled;
    @Value("${llm.claude.enabled:false}")
    private boolean claudeEnabled;

    @Value("${jsonFilePath}")
    String jsonFilePath ;
    @Value("${textFilePath}")
    String textFilePath  ;

    @Value("${temperature:0.5f}")
    float temperature ;

    private final CloseableHttpClient httpClient = HttpClients.createDefault();
    private final Gson gson = new Gson(); // Gson for JSON operations
    private List<DocumentVector> vectorStore = new ArrayList<>();
    //private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();


    private void loadVectorStoreFromJson(String filePath) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            vectorStore = gson.fromJson(br, new TypeToken<List<DocumentVector>>(){}.getType());
        }
    }

    private void generateVectorStoreFromText(String filePath) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    List<Float> embedding = getEmbedding(line);
                    vectorStore.add(new DocumentVector(line, embedding));
                }
            }
        }
    }

    private void saveVectorStoreToJson(String filePath) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(filePath))) {
            gson.toJson(vectorStore, bw);
        }
    }
    @Autowired
    Environment env;
    public Map compareLlmResponses(boolean ragEnabled, String query) {
        try {

            Map<String, Object> response = new HashMap<>();
            response.put("question", query);

            String context = "";
            if(ragEnabled) {
                List<Float> queryEmbedding = getEmbedding(query);
                List<String> retrievedDocs = queryVectorStore(queryEmbedding);
                context = String.join("\n", retrievedDocs);
                response.put("context", context);
            }

            if (openAiEnabled) {

                long start = System.currentTimeMillis();
                String openAiResponse =queryOpenAI(query, context);
                double timeTaken = (System.currentTimeMillis() - start)/1000;
                response.put("OpenAI", new TimedResponse(openAiResponse,  timeTaken));
            }
            if (geminiEnabled) {
                long start = System.currentTimeMillis();
                String geminiResponse = queryGemini(query, context);
                double timeTaken = (System.currentTimeMillis() - start)/1000;
                response.put("Gemini", new TimedResponse(geminiResponse, timeTaken));
            }
            if (claudeEnabled) {
                long start = System.currentTimeMillis();
                String claudeResponse = queryClaude(query, context);
                double timeTaken = (System.currentTimeMillis() - start)/1000;
                response.put("Claude", new TimedResponse(claudeResponse, timeTaken));
            }
            return response;


        } catch (IOException e) {
            throw new RuntimeException("Error processing RAG request: " + e.getMessage(), e);
        }
    }

    private List<Float> getEmbedding(String text) throws IOException {
        HttpPost request = new HttpPost(OPENAI_API_URL + "/embeddings");
        String json = gson.toJson(new EmbeddingRequest("text-embedding-ada-002", text));
        request.setEntity(new StringEntity(json, StandardCharsets.UTF_8));
        request.setHeader("Content-Type", "application/json");
        request.setHeader("Authorization", "Bearer " + OPENAI_API_KEY);

        String response = httpClient.execute(request, resp -> new String(resp.getEntity().getContent().readAllBytes()));
        JsonObject jsonObject = gson.fromJson(response, JsonObject.class);
        JsonArray dataArray = jsonObject.getAsJsonArray("data");
        return gson.fromJson(dataArray.get(0).getAsJsonObject().get("embedding"), new TypeToken<List<Float>>() {
        }.getType());
    }

    private List<String> queryVectorStore(List<Float> queryEmbedding) {
        return vectorStore.stream()
                .sorted(Comparator.comparingDouble(doc -> -cosineSimilarity(doc.embedding, queryEmbedding)))
                .limit(3)
                .map(doc -> doc.text)
                .toList();
    }

    private double cosineSimilarity(List<Float> v1, List<Float> v2) {
        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        for (int i = 0; i < v1.size(); i++) {
            dotProduct += v1.get(i) * v2.get(i);
            norm1 += Math.pow(v1.get(i), 2);
            norm2 += Math.pow(v2.get(i), 2);
        }
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    private String queryOpenAI(String query, String context) throws IOException {
        HttpPost request = new HttpPost(OPENAI_API_URL + "/chat/completions");
        String json = gson.toJson(new OpenAIRequest("gpt-4o", "Context: " + context + "\nQuery: " + query, temperature));
        request.setEntity(new StringEntity(json, StandardCharsets.UTF_8));
        request.setHeader("Content-Type", "application/json");
        request.setHeader("Authorization", "Bearer " + OPENAI_API_KEY);

        String response = httpClient.execute(request, resp -> new String(resp.getEntity().getContent().readAllBytes()));
        JsonObject jsonObject = gson.fromJson(response, JsonObject.class);
        return jsonObject.getAsJsonArray("choices").get(0).getAsJsonObject()
                .get("message").getAsJsonObject()
                .get("content").getAsString();
    }

    private String queryGemini(String query, String context) throws IOException {

        HttpPost request = new HttpPost(GEMINI_API_URL + "?key=" + GEMINI_API_KEY);
        String json = gson.toJson(new GeminiRequest("Context: " + context + "\nQuery: " + query));
        request.setEntity(new StringEntity(json, StandardCharsets.UTF_8));
        request.setHeader("Content-Type", "application/json");

        String response = httpClient.execute(request, resp -> new String(resp.getEntity().getContent().readAllBytes()));
        JsonObject jsonObject = gson.fromJson(response, JsonObject.class);
        return jsonObject.getAsJsonArray("candidates").get(0).getAsJsonObject()
                .get("content").getAsJsonObject()
                .getAsJsonArray("parts").get(0).getAsJsonObject()
                .get("text").getAsString();
    }

    private String queryClaude(String query, String context) throws IOException {
        HttpPost request = new HttpPost(CLAUDE_API_URL);
        String json = gson.toJson(new ClaudeRequest("claude-3-7-sonnet-20250219", temperature, query, context));

        request.setEntity(new StringEntity(json, StandardCharsets.UTF_8));
        request.setHeader("Content-Type", "application/json");
        request.setHeader("x-api-key", CLAUDE_API_KEY);
        request.setHeader("anthropic-version", "2023-06-01");

        String response = httpClient.execute(request, resp -> new String(resp.getEntity().getContent().readAllBytes()));
        JsonObject jsonObject = gson.fromJson(response, JsonObject.class);

        return jsonObject.getAsJsonArray("content").get(0).getAsJsonObject().get("text").getAsString();

    }

    // Helper classes for JSON requests
    public record EmbeddingRequest(String model, String input) {}


    record OpenAIRequest (String model, List<Message> messages, float temperature) {

        public OpenAIRequest(String model, String content, float temperature) {
            this(model, List.of(new Message("user", content)), temperature);
        }

        record Message(String role, String content) {}
    }


    record GeminiRequest (List<Content> contents){

        public GeminiRequest(String text) {
            this(List.of(new Content(List.of(new Part(text)))));
        }
        record Content(List<Part> parts) {}
        record Part(String text) {}

    }

    record ClaudeRequest(String model,
                         int max_tokens,// = 200,
                         float temperature,
                         String system,
                         List<UserMessage> messages) {

        public ClaudeRequest(String model, float temperature, String question, String context) {

            this(model, 200, temperature,

                    "SYSTEM_PROMPT=\"You are a helpful assistant. Use the following context from the knowledge base to answer the user's question. \n" +
                            "If you cannot answer the question with the context provided, say that you don't know.\n" +
                            "\n" +
                            "Context from knowledge base:\n" +
                            context, List.of(new UserMessage("user", "'" + question + "'")));

        }

        public record UserMessage(String role, String content) {}

    }

    public record DocumentVector(String text, List<Float> embedding) {}

    public record TimedResponse(String response, double timeMs) {}

    @PostConstruct
    public void init() throws IOException {
        File jsonFile = new File(jsonFilePath);

        if (jsonFile.exists()) {

            loadVectorStoreFromJson(jsonFilePath);
        } else {

            generateVectorStoreFromText(textFilePath);
            saveVectorStoreToJson(jsonFilePath);
        }

    }
}
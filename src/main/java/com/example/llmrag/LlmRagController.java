package com.example.llmrag;

import com.example.llmrag.service.LlmRagService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
public class LlmRagController {

    @Autowired
    private LlmRagService llmRagService;

    @GetMapping("/rag/compare")
    public Map<String,Object> compareLlmResponses(@RequestParam(name = "ragEnabled", defaultValue = "true")   boolean ragEnabled, @RequestParam String query) {
        return llmRagService.compareLlmResponses(ragEnabled,query);
    }
}
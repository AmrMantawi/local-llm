#pragma once

#include "llm.h"
#include <string>
#include <vector>
#include <functional>
#include <rkllm.h>

/// RKNN-LLM-based LLM adapter for Rockchip NPU.
class RknnLLM : public ILLM {
public:
    /// Initialize RKLLM
    bool init() override;

    /// Generate a response to the input prompt
    bool generate(const std::string &prompt, std::string &response) override;

    /// Async generate with text chunk callback for TTS streaming
    bool generate_async(const std::string &prompt, std::string &response, 
                       std::function<void(const std::string&)> callback) override;

    /// Release resources
    void shutdown() override;

private:
    // RKNN LLM handle
    LLMHandle handle = nullptr;
    
    // RKNN LLM parameters
    RKLLMParam param;
    
    // Chat configuration
    const std::string chat_symb = ":";
    std::vector<std::string> antiprompts = {"Finn:"};
    
    // Context management
    int max_context_len = 4096;
    int max_new_tokens = 512;
    bool keep_history = true;
    
    // Async generation state
    std::string current_response;
    std::function<void(const std::string&)> async_callback;
    std::string token_buffer;  // Buffer to accumulate tokens for chunking
    int word_count = 0;  // Counter for words in current chunk
    bool in_word = false; // Tracks if we're currently inside a word across callbacks
    
    // Internal callback function for RKNN LLM
    static int rknn_callback(RKLLMResult* result, void* userdata, LLMCallState state);
};

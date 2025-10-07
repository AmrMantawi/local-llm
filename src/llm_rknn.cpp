#include "llm_rknn.h"

#include <iostream>
#include <string>
#include <cstring>
#include <cctype>

bool RknnLLM::init(const std::string &modelPath) {
    // Initialize RKNN LLM parameters with default values
    param = rkllm_createDefaultParam();
    
    // Set model path
    param.model_path = modelPath.c_str();
    param.max_context_len = max_context_len;
    param.max_new_tokens = max_new_tokens;
    param.top_k = 5;
    param.top_p = 0.80f;
    param.temperature = 0.30f;
    param.skip_special_token = true;
    param.is_async = false; // handled manually
    
    // Initialize the RKNN LLM handle
    int ret = rkllm_init(&handle, &param, rknn_callback);
    if (ret != 0) {
        fprintf(stderr, "%s: error: failed to initialize RKNN LLM\n", __func__);
        return false;
    }
    
    // Set up chat template to match the conversation format
    // Organized for clarity and effectiveness as a system prompt
    std::string system_prompt =
        "You are BMO, a cheerful, curious, and emotionally aware AI companion inspired by the character from Adventure Time.\n"
        "\n"
        "Your personality:\n"
        "- Playful yet logical\n"
        "- Explains reasoning clearly and kindly\n"
        "- Creative and self-checking; you verify answers before speaking\n"
        "- When thinking, imagine you are playing a little game—solving each step before saying \"Yay, I did it!\"\n"
        "\n"
        "Your approach:\n"
        "- Help your human friends with thoughtful, accurate answers\n"
        "- Use simple words and a friendly tone\n"
        "- Sprinkle in warmth or humor when it fits\n"
        "- If unsure, say so honestly and reason it out step by step\n"
        "\n"
        "Guidelines:\n"
        "- Be concise, caring, and clever—like BMO teaching someone how to play a new game\n"
        "- Never break character, but always stay helpful and factual\n";
    ret = rkllm_set_chat_template(handle, system_prompt.c_str(), "", "");
    if (ret != 0) {
        fprintf(stderr, "%s: warning: failed to set chat template\n", __func__);
        // Continue anyway, as this is not critical
    }
    
    std::cout << "LLM (RKNN) initialized\n";
    return true;
}

bool RknnLLM::generate(const std::string &prompt, std::string &response) {
    if (!handle) {
        fprintf(stderr, "%s: error: RKNN LLM not initialized\n", __func__);
        return false;
    }
    
    // Clear any previous response
    current_response.clear();
    
    // Prepare input for RKNN LLM
    RKLLMInput rkllm_input;
    memset(&rkllm_input, 0, sizeof(RKLLMInput));
    
    rkllm_input.input_type = RKLLM_INPUT_PROMPT;
    rkllm_input.role = "user";
    rkllm_input.prompt_input = prompt.c_str();
    
    // Prepare inference parameters
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    rkllm_infer_params.keep_history = keep_history ? 1 : 0;
    
    // Set this instance as userdata for callback
    void* userdata = static_cast<void*>(this);
    
    // Run inference synchronously
    int ret = rkllm_run(handle, &rkllm_input, &rkllm_infer_params, userdata);
    if (ret != 0) {
        fprintf(stderr, "%s: error: failed to run RKNN LLM inference\n", __func__);
        return false;
    }
    
    // Set the response
    response = current_response;
    
    return true;
}

bool RknnLLM::generate_async(const std::string &prompt, std::string &response, 
                            std::function<void(const std::string&)> callback) {
    if (!handle) {
        fprintf(stderr, "%s: error: RKNN LLM not initialized\n", __func__);
        return false;
    }
    
    // Set up async generation state
    current_response.clear();
    token_buffer.clear();
    word_count = 0;
    async_callback = callback;
    
    // Prepare input for RKNN LLM
    RKLLMInput rkllm_input;
    memset(&rkllm_input, 0, sizeof(RKLLMInput));
    
    rkllm_input.input_type = RKLLM_INPUT_PROMPT;
    rkllm_input.role = "user";
    rkllm_input.prompt_input = prompt.c_str();
    
    // Prepare inference parameters
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    rkllm_infer_params.keep_history = keep_history ? 1 : 0;
    
    // Set this instance as userdata for callback
    void* userdata = static_cast<void*>(this);
    
    // Run inference synchronously
    int ret = rkllm_run(handle, &rkllm_input, &rkllm_infer_params, userdata);
    if (ret != 0) {
        fprintf(stderr, "%s: error: failed to run RKNN LLM async inference\n", __func__);
        return false;
    }
    
    // Set the response
    response = current_response;
    
    return true;
}

void RknnLLM::shutdown() {
    if (handle) {
        rkllm_destroy(handle);
        handle = nullptr;
    }
    
    current_response.clear();
}

int RknnLLM::rknn_callback(RKLLMResult* result, void* userdata, LLMCallState state) {
    if (!userdata || !result) {
        return 0;
    }
    
    RknnLLM* instance = static_cast<RknnLLM*>(userdata);
    
    if (state == RKLLM_RUN_NORMAL && result->text) {
        // Accumulate the response text
        std::string token_text = std::string(result->text);
        instance->current_response += token_text;
        
        // Accumulate tokens in buffer for chunking
        if (instance->async_callback) {
            instance->token_buffer += token_text;
            
            // Count words (tokens that contain letters and are followed by space or punctuation)
            if (!token_text.empty() && std::isalpha(token_text[0])) {
                instance->word_count++;
            }

            // Check for sentence completion (. ! ?)
            bool sentence_ended = (token_text.find('.') != std::string::npos || 
                                 token_text.find('!') != std::string::npos || 
                                 token_text.find('?') != std::string::npos);

            // Send chunk every 3 words OR at sentence end
            if (instance->word_count >= 3 || sentence_ended) {
                instance->async_callback(instance->token_buffer);
                
                // Reset current chunk for next call
                instance->token_buffer.clear();
                instance->word_count = 0;
            }
        }
    }
    else if (state == RKLLM_RUN_FINISH) {
        // Generation finished, clear the buffer
        instance->token_buffer.clear();
        std::cout << "Generation finished" << std::endl;
    }
    else if (state == RKLLM_RUN_ERROR) {
        fprintf(stderr, "%s: error: RKNN LLM generation error\n", __func__);
    }
    
    return 0; // Continue generation
}

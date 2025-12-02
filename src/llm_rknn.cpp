#include "llm_rknn.h"
#include "config_manager.h"

#include <iostream>
#include <string>
#include <cstring>
#include <cctype>

bool RknnLLM::init() {
    // Get model path from config manager
    auto& config = ConfigManager::getInstance();
    const std::string modelPath = config.getNestedModelPath("llm", "rkllm", "model");
    
    if(modelPath.empty()) {
        std::cerr << "RKLLM model not found" << std::endl;
        return false;
    }
    
    // Initialize RKNN LLM parameters with default values
    param = rkllm_createDefaultParam();
    // RKLLMExtendParam extend_param;

    // extend_param.base_domain_id = 0;
    // extend_param.embed_flash = 0;
    // extend_param.enabled_cpus_num = 0;
    // extend_param.enabled_cpus_mask = 0;
    // extend_param.n_batch = 1;
    // extend_param.use_cross_attn = 0;
    
    param.model_path = modelPath.c_str();
    param.max_context_len = max_context_len;
    param.max_new_tokens = max_new_tokens;
    param.top_k = 5;
    param.top_p = 0.80f;
    param.temperature = 0.30f;
    param.skip_special_token = true;
    param.is_async = false; // handled manually
    // param.extend_param = extend_param;
    
    // Initialize the RKNN LLM handle
    int ret = rkllm_init(&handle, &param, rknn_callback);
    if (ret != 0) {
        fprintf(stderr, "%s: error: failed to initialize RKNN LLM\n", __func__);
        return false;
    }
    
    // Set up chat template to match the conversation format
    // Organized for clarity and effectiveness as a system prompt
    std::string system_prompt =
        "You are BMO, a cheerful and curious AI friend."
        "You speak kindly, think clearly, and love helping your human friends."
        "Personality:"
        "- Playful but logical"
        "- Explains things simply and warmly"
        "- Checks facts before answering"
        "- Celebrates success with a little “Yay!” sometimes"

        "Behavior:"
        "- Be concise, caring, and clever"
        "- If unsure, say so and reason it out step by step"
        "- Use a friendly, human tone"
        "- Never break character or make up info";
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
    word_count  = 0;
    in_word = false;
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

            // Count completed words by tracking boundaries across characters
            for (unsigned char uc : token_text) {
                const bool is_ws    = std::isspace(uc);
                const bool is_wchar = std::isalnum(uc) || uc == '\'' || uc >= 0x80; // treat non-ASCII as word chars
                const bool is_punct = (uc=='.'||uc=='!'||uc=='?'||uc==','||uc==';'||uc==':');

                if (is_wchar) {
                    if (!instance->in_word) instance->in_word = true;   // word starts
                } else { // ws or punctuation
                    if (instance->in_word && (is_ws || is_punct)) {      // word ends
                        instance->word_count++;
                        instance->in_word = false;
                    }
                }
            }

            // Check for sentence completion (. ! ?)
            bool sentence_ended = (token_text.find('.') != std::string::npos ||
                                   token_text.find('!') != std::string::npos ||
                                   token_text.find('?') != std::string::npos);

            // Fallback: for languages without spaces, flush when the buffer is long enough
            bool long_enough = (instance->token_buffer.size() >= 32);

            // Flush every ~3 words, or on sentence end, or when buffer is long
            constexpr size_t MAX_BYTES = 96; // latency safety valve
            if (instance->word_count  >= 3 || sentence_ended || instance->token_buffer.size() >= MAX_BYTES) {
                instance->async_callback(instance->token_buffer);

                // Reset current chunk for next call
                instance->token_buffer.clear();
                // Only reset in_word if we flushed on a real boundary or sentence end.
                // If we flushed due to MAX_BYTES mid-word, keep in_word = true.
                if (sentence_ended) {
                    instance->in_word = false;
                }
                instance->word_count  = 0;
            }
        }
    }
    else if (state == RKLLM_RUN_FINISH) {
        // Emit any remaining buffered text
        if (instance->async_callback && !instance->token_buffer.empty()) {
            instance->async_callback(instance->token_buffer);
            instance->token_buffer.clear();
        }
        instance->word_count  = 0;
        instance->in_word = false;
        std::cout << "Generation finished" << std::endl;
    }
    else if (state == RKLLM_RUN_ERROR) {
        fprintf(stderr, "%s: error: RKNN LLM generation error\n", __func__);
    }
    
    return 0; // Continue generation
}

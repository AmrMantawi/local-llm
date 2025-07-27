#pragma once

#include "llm.h"
#include <string>
#include <vector>
#include "llama.h"

/// llama.cpp-based LLM adapter.
class LlamaLLM : public ILLM {
public:
    /// Initialize the LLM with a given model path
    bool init(const std::string &modelPath) override;

    /// Generate a response to the input prompt
    bool generate(const std::string &prompt, std::string &response) override;

    /// Release resources
    void shutdown() override;

private:

    // text inference variables
    int ngl = 0;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_sampler * smpl = nullptr;
    llama_model * model = nullptr;
    std::string path_session = "";
    bool need_to_save_session = false; // Number of matching tokens in the session
    std::vector<llama_token> embd_inp; // Input tokens for the model
    std::vector<llama_token> embd;
    llama_batch batch;

    std::vector<llama_token> session_tokens; // Tokens from the session
    const std::string chat_symb = ":";

    int n_keep = 0;
    int n_ctx = 2048;
    int n_past = 0;
    int n_prev = 64; // TODO: make configurable
    int n_session_consumed = 0;

    std::vector<std::string> antiprompts = {"Finn:"};

};
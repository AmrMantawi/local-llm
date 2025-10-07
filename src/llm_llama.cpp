#include "llm_llama.h"
#include "common-sdl.h"
#include "common.h"

#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#include <cctype>

// Prompt template for the conversation - defines the chat format and personality
const std::string k_prompt_llama = R"(Text transcript of a never ending dialog, where {0} interacts with an AI assistant named {1}.
{1} is helpful, kind, honest, friendly, good at writing and never fails to answer {0}'s requests immediately and with details and precision.
There are no annotations like (30 seconds passed...) or (to himself), just what {0} and {1} say aloud to each other.
The transcript only includes text, it does not include markup like HTML and Markdown.
{1} responds with short and concise answers.

{0}{4} Hello, {1}!
{1}{4} Hello {0}! How may I help you today?
{0}{4} What time is it?
{1}{4} It is {2} o'clock.
{0}{4} What year is it?
{1}{4} We are in {3}.
{0}{4} What is a cat?
{1}{4} A cat is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae.
{0}{4} Name a color.
{1}{4} Blue
{0}{4})";

// Set number of threads for processing
int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());

// Convert a llama token to its text representation
static std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Allocate buffer for token text (start with 8 chars, will resize if needed)
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, false);
    if (n_tokens < 0) {
        // Negative value means buffer was too small, resize and try again
        result.resize(-n_tokens);
        int check = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, false);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return std::string(result.data(), result.size());
}

// Convert text to llama tokens (tokenization)
static std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Estimate token count (text length + 1 if adding BOS token)
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_bos, false);
    if (n_tokens < 0) {
        // Negative value means buffer was too small, resize and try again
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_bos, false);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

bool LlamaLLM::init(const std::string &modelPath) {
    // Initialize llama backend
    llama_backend_init();

    // Load the model with GPU layers configuration
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl; // Number of GPU layers (0 = CPU only)

    model = llama_model_load_from_file(modelPath.c_str(), model_params);
    if (!model) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return false;
    }

    // Get vocabulary from the model
    vocab = llama_model_get_vocab(model);

    // Initialize the context for inference
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;      // Context window size
    ctx_params.n_batch = n_ctx;    // Batch size for processing

    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return false;
    }

    // Build the initial prompt by replacing placeholders with actual values
    std::string prompt_llama = k_prompt_llama;

    // Add leading space (required for tokenization)
    prompt_llama.insert(0, 1, ' ');

    // Replace placeholders with actual values
    prompt_llama = ::replace(prompt_llama, "{0}", "Finn");  // User name
    prompt_llama = ::replace(prompt_llama, "{1}", "BMO");   // Bot name

    // Add current time
    {
        std::string time_str;
        {
            time_t t = time(0);
            struct tm * now = localtime(&t);
            char buf[128];
            strftime(buf, sizeof(buf), "%H:%M", now);
            time_str = buf;
        }
        prompt_llama = ::replace(prompt_llama, "{2}", time_str);
    }

    // Add current year
    {
        std::string year_str;
        {
            time_t t = time(0);
            struct tm * now = localtime(&t);
            char buf[128];
            strftime(buf, sizeof(buf), "%Y", now);
            year_str = buf;
        }
        prompt_llama = ::replace(prompt_llama, "{3}", year_str);
    }

    // Add chat symbol
    prompt_llama = ::replace(prompt_llama, "{4}", chat_symb);

    // Print the initial prompt for debugging
    printf("prompt: %s\n", prompt_llama.c_str());

    // Initialize batch for token processing
    batch = llama_batch_init(llama_n_ctx(ctx), 0, 1);

    // Initialize the sampler with temperature, top-k, and top-p parameters
    const float top_k = 5;      // Number of top tokens to consider
    const float top_p = 0.80f;  // Nucleus sampling parameter
    const float temp  = 0.30f;  // Temperature for randomness

    const int seed = 0;

    auto sparams = llama_sampler_chain_default_params();
    smpl = llama_sampler_chain_init(sparams);

    // Configure sampling strategy based on temperature
    if (temp > 0.0f) {
        // Use temperature-based sampling for more creative responses
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp (temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist (seed));
    } else {
        // Use greedy sampling for deterministic responses
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    // Tokenize the initial prompt
    embd_inp = ::llama_tokenize(ctx, prompt_llama, true);

    // Session management (currently disabled to avoid debug messages)
    if (!path_session.empty()) {
        fprintf(stderr, "%s: attempting to load saved session from %s\n", __func__, path_session.c_str());
  
        // Check if session file exists
        FILE * fp = std::fopen(path_session.c_str(), "rb");
        if (fp != NULL) {
            std::fclose(fp);

            // Load session tokens
            session_tokens.resize(llama_n_ctx(ctx));
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            for (size_t i = 0; i < session_tokens.size(); i++) {
                embd_inp[i] = session_tokens[i];
            }

            fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
        } else {
            fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
        }
    }

    // Evaluate the initial prompt to set up the context
    printf("\n");
    printf("%s : initializing - please wait ...\n", __func__);

    // Prepare batch for initial prompt evaluation
    {
        batch.n_tokens = embd_inp.size();
        printf("%s : evaluating initial prompt with %u tokens\n", __func__, batch.n_tokens);
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.token[i]     = embd_inp[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = i == batch.n_tokens - 1;  // Only compute logits for last token
        }
    }

    // Decode the initial prompt
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
        return false;
    }

    // Set up session management flags
    need_to_save_session = !path_session.empty();
    
    // Initialize context tracking variables
    n_keep   = embd_inp.size();  // Number of tokens to keep when context is full
    n_ctx    = llama_n_ctx(ctx);  // Total context size

    n_past = n_keep;  // Current position in context
    n_session_consumed = !path_session.empty() && session_tokens.size() > 0 ? session_tokens.size() : 0;
    
    std::cout << "LLM (Llama) initialized\n";

    return true;
}

bool LlamaLLM::generate(const std::string &prompt, std::string &response) {
    // Tokenize the user's input prompt
    const std::vector<llama_token> tokens = llama_tokenize(ctx, prompt.c_str(), false);
    if (prompt.empty() || tokens.empty()) {
        response = "";
        return true;
    }
    
    // Format the input for the model: add space prefix and bot response format
    std::string formatted_text = " " + prompt;
    formatted_text += "\nBMO" + chat_symb;

    // Tokenize the formatted input
    embd = ::llama_tokenize(ctx, formatted_text, false);

    // Update session tokens if session saving is enabled
    if (!path_session.empty()) {
        session_tokens.insert(session_tokens.end(), tokens.begin(), tokens.end());
    }

    // Main text generation loop
    bool done = false;
    std::string text_to_speak;
    
    while (true) {
        // Process input tokens if we have any
        if (embd.size() > 0) {
            // Check if we're running out of context window
            if (n_past + (int) embd.size() > n_ctx) {
                // Reset context and keep only the most recent tokens
                n_past = n_keep;

                // Insert recent tokens at the start to maintain some context
                embd.insert(embd.begin(), embd_inp.begin() + embd_inp.size() - n_prev, embd_inp.end());
                // Disable session saving if context is full
                path_session = "";
            }

            // Try to reuse matching tokens from saved session to avoid recomputation
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        // Mismatch found, truncate session
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        i++;
                        break;
                    }
                }
                if (i > 0) {
                    // Remove processed tokens from input
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            // Update session with new tokens
            if (embd.size() > 0 && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }

            // Prepare batch for decoding
            {
                batch.n_tokens = embd.size();

                for (int i = 0; i < batch.n_tokens; i++) {
                    batch.token[i]     = embd[i];
                    batch.pos[i]       = n_past + i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i]    = i == batch.n_tokens - 1;  // Only compute logits for last token
                }
            }

            // Decode the tokens through the model
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "%s : failed to decode\n", __func__);
                return false;
            }
        }

        // Add processed tokens to input history
        embd_inp.insert(embd_inp.end(), embd.begin(), embd.end());
        n_past += embd.size();

        // Clear processed tokens
        embd.clear();

        // Exit if generation is complete
        if (done) break;

        {
            // Generate next token using the sampler

            // Save session state if needed
            if (!path_session.empty() && need_to_save_session) {
                need_to_save_session = false;
                llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
            }

            // Sample next token from the model
            const llama_token id = llama_sampler_sample(smpl, ctx, -1);

            if (id != llama_vocab_eos(vocab)) {
                // Add the new token to the generation
                embd.push_back(id);

                // Convert token to text and add to response
                std::string token_text = llama_token_to_piece(ctx, id);
                text_to_speak += token_text;

                // Don't print here - let main.cpp handle the output
                // printf("%s", token_text.c_str());
                // fflush(stdout);
            }
        }

        {
            // Check for antiprompts (stop conditions) in the recent output
            std::string last_output;
            // Look at the last 16 tokens to check for stop conditions
            for (int i = embd_inp.size() - 16; i < (int) embd_inp.size(); i++) {
                if (i >= 0) {
                    last_output += llama_token_to_piece(ctx, embd_inp[i]);
                }
            }
            if (!embd.empty()) {
                last_output += llama_token_to_piece(ctx, embd[0]);
            }

            // Check if any antiprompt patterns are found
            for (std::string & antiprompt : antiprompts) {
                if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
                    done = true;
                    // Remove the antiprompt from the response
                    text_to_speak = ::replace(text_to_speak, antiprompt, "");
                    // fflush(stdout); // Not needed since we're not printing here
                    need_to_save_session = true;
                    break;
                }
            }
        }
    }

    // Set the final response
    response = text_to_speak;
    
    return true;
}

bool LlamaLLM::generate_async(const std::string &prompt, std::string &response, 
                             std::function<void(const std::string&)> callback) {
    // Tokenize the user's input prompt
    const std::vector<llama_token> tokens = llama_tokenize(ctx, prompt.c_str(), false);
    if (prompt.empty() || tokens.empty()) {
        response = "";
        return true;
    }
    
    // Format the input for the model: add space prefix and bot response format
    std::string formatted_text = " " + prompt;
    formatted_text += "\nBMO" + chat_symb;

    // Tokenize the formatted input
    embd = ::llama_tokenize(ctx, formatted_text, false);

    // Update session tokens if session saving is enabled
    if (!path_session.empty()) {
        session_tokens.insert(session_tokens.end(), tokens.begin(), tokens.end());
    }

    // Main text generation loop with chunk-level streaming
    bool done = false;
    std::string text_to_speak;
    std::string token_buffer;
    int word_count = 0;
    bool in_word = false;
    
    while (true) {
        // Process input tokens if we have any
        if (embd.size() > 0) {
            // Check if we're running out of context window
            if (n_past + (int) embd.size() > n_ctx) {
                // Reset context and keep only the most recent tokens
                n_past = n_keep;

                // Insert recent tokens at the start to maintain some context
                embd.insert(embd.begin(), embd_inp.begin() + embd_inp.size() - n_prev, embd_inp.end());
                // Disable session saving if context is full
                path_session = "";
            }

            // Try to reuse matching tokens from saved session to avoid recomputation
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        // Mismatch found, truncate session
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        i++;
                        break;
                    }
                }
                if (i > 0) {
                    // Remove processed tokens from input
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            // Update session with new tokens
            if (embd.size() > 0 && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }

            // Prepare batch for decoding
            {
                batch.n_tokens = embd.size();

                for (int i = 0; i < batch.n_tokens; i++) {
                    batch.token[i]     = embd[i];
                    batch.pos[i]       = n_past + i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i]    = i == batch.n_tokens - 1;  // Only compute logits for last token
                }
            }

            // Decode the tokens through the model
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "%s : failed to decode\n", __func__);
                return false;
            }
        }

        // Add processed tokens to input history
        embd_inp.insert(embd_inp.end(), embd.begin(), embd.end());
        n_past += embd.size();

        // Clear processed tokens
        embd.clear();

        // Exit if generation is complete
        if (done) break;

        {
            // Generate next token using the sampler

            // Save session state if needed
            if (!path_session.empty() && need_to_save_session) {
                need_to_save_session = false;
                llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
            }

            // Sample next token from the model
            const llama_token id = llama_sampler_sample(smpl, ctx, -1);

            if (id != llama_vocab_eos(vocab)) {
                // Add the new token to the generation
                embd.push_back(id);

                // Convert token to text and add to response
                std::string token_text = llama_token_to_piece(ctx, id);
                text_to_speak += token_text;
                token_buffer += token_text;

                // Count completed words by tracking boundaries across characters
                for (unsigned char uc : token_text) {
                    const bool is_ws    = std::isspace(uc);
                    const bool is_wchar = std::isalnum(uc) || uc == '\'' || uc >= 0x80; // treat non-ASCII as word chars
                    const bool is_punct = (uc=='.'||uc=='!'||uc=='?'||uc==','||uc==';'||uc==':');

                    if (is_wchar) {
                        if (!in_word) in_word = true;   // word starts
                    } else { // ws or punctuation
                        if (in_word && (is_ws || is_punct)) { // word ends
                            word_count++;
                            in_word = false;
                        }
                    }
                }

                // Check for sentence completion (. ! ?)
                bool sentence_ended = (token_text.find('.') != std::string::npos ||
                                       token_text.find('!') != std::string::npos ||
                                       token_text.find('?') != std::string::npos);

                // Flush every ~3 words, or on sentence end, or when buffer is long
                constexpr size_t MAX_BYTES = 96; // latency safety valve
                if (word_count >= 4 || sentence_ended || token_buffer.size() >= MAX_BYTES) {
                    callback(token_buffer);

                    // Reset current chunk for next call
                    token_buffer.clear();
                    // Only reset in_word if we flushed on a sentence end. If we flushed
                    // due to MAX_BYTES mid-word, keep in_word = true.
                    if (sentence_ended) {
                        in_word = false;
                    }
                    word_count = 0;
                }
            }
        }

        {
            // Check for antiprompts (stop conditions) in the recent output
            std::string last_output;
            // Look at the last 16 tokens to check for stop conditions
            for (int i = embd_inp.size() - 16; i < (int) embd_inp.size(); i++) {
                if (i >= 0) {
                    last_output += llama_token_to_piece(ctx, embd_inp[i]);
                }
            }
            if (!embd.empty()) {
                last_output += llama_token_to_piece(ctx, embd[0]);
            }

            // Check if any antiprompt patterns are found
            for (std::string & antiprompt : antiprompts) {
                if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
                    done = true;
                    // Remove the antiprompt from the response
                    text_to_speak = ::replace(text_to_speak, antiprompt, "");
                    need_to_save_session = true;
                    break;
                }
            }
        }
    }

    // Handle any remaining text that hasn't been sent yet
    if (!token_buffer.empty()) {
        callback(token_buffer);
        token_buffer.clear();
    }

    // Set the final response
    response = text_to_speak;
    
    return true;
}

void LlamaLLM::shutdown() {
    // Free the sampler
    llama_sampler_free(smpl);
    
    // Free the batch
    llama_batch_free(batch);
    
    // Free the context
    llama_free(ctx);

    // Shutdown the llama backend
    llama_backend_free();
}

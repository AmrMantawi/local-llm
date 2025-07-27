#include "stt_whisper.h"

#include <iostream>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Precompiled regex patterns for cleaning transcription output
static const std::regex RE_SQUARE_BRACKETS("\\[.*?\\]");
static const std::regex RE_PARENS("\\(.*?\\)");
static const std::regex RE_NON_ALPHANUMERIC("[^a-zA-Z0-9\\.,\\?!\\s\\:\\'\\-]");
static const std::regex RE_WS_LEADING("^\\s+");
static const std::regex RE_WS_TRAILING("\\s+$");

const int32_t WhisperSTT::N_THREADS = std::min(4, static_cast<int32_t>(std::thread::hardware_concurrency()));

// trim whitespace from both ends of a string
static std::string trim(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && isspace(str[start])) {
        start += 1;
    }
    while (end > start && isspace(str[end - 1])) {
        end -= 1;
    }
    return str.substr(start, end - start);
}

bool WhisperSTT::init(const std::string &modelPath) {
  struct whisper_context_params cparams = whisper_context_default_params();

  cparams.use_gpu    = false; // use GPU acceleration
  cparams.flash_attn = false; // use flash attention

  ctx = whisper_init_from_file_with_params(modelPath.c_str(), cparams);
  if (!ctx) {
      fprintf(stderr, "No whisper.cpp model specified.\n");
      return false;
  }

  std::cout << "STT (Whisper) initialized\n";
  return true;
}

bool WhisperSTT::transcribe(const std::vector<float> &pcmf32, std::string &outText) {
  // const auto t_start = std::chrono::high_resolution_clock::now();
  // prob = 0.0f;
  // t_ms = 0;

  std::vector<whisper_token> prompt_tokens;

  whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

  prompt_tokens.resize(1024);
  prompt_tokens.resize(whisper_tokenize(ctx, "", prompt_tokens.data(), prompt_tokens.size()));

  // Configure Whisper inference parameters
  wparams.print_progress   = false; // print progress
  wparams.print_special    = false; // print special tokens
  wparams.print_realtime   = false; // print realtime transcription
  wparams.print_timestamps = false; // print timestamps for each text segment
  wparams.translate        = false; // translate to English
  wparams.no_context       = true; // do not use past transcription (if any) as initial prompt for the decoder. TODO: setup
  wparams.single_segment   = true; // force single segment output (useful for streaming)
  wparams.max_tokens       = MAX_TOKENS; // max tokens to use from past text as prompt for the decoder
  wparams.language         = "en"; // language to use for the decoder  
  wparams.n_threads        = N_THREADS; // number of threads to use for processing
  wparams.prompt_tokens    = prompt_tokens.empty() ? nullptr : prompt_tokens.data();
  wparams.prompt_n_tokens  = prompt_tokens.empty() ? 0       : prompt_tokens.size();
  wparams.audio_ctx        = 0;

  // Run Whisper transcription
  if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) 
  {
    return false;
  }

  // int prob_n = 0;

  // Collect transcribed segments into a single string
  std::string all_heard;
  const int n_segments = whisper_full_n_segments(ctx);
  for (int i = 0; i < n_segments; ++i) {
      const char * text = whisper_full_get_segment_text(ctx, i);

      all_heard += text;

      // const int n_tokens = whisper_full_n_tokens(ctx, i);
      // for (int j = 0; j < n_tokens; ++j) {
      //     const auto token = whisper_full_get_token_data(ctx, i, j);

      //     prob += token.p;
      //     ++prob_n;
      // }
  }

  std::string text_heard = ::trim(all_heard);

  // Clean and normalize the text
  text_heard = std::regex_replace(text_heard, RE_SQUARE_BRACKETS, "");
  text_heard = std::regex_replace(text_heard, RE_PARENS, "");
  text_heard = std::regex_replace(text_heard, RE_NON_ALPHANUMERIC, "");
  text_heard = text_heard.substr(0, text_heard.find_first_of('\n'));
  text_heard = std::regex_replace(text_heard, RE_WS_LEADING, "");
  text_heard = std::regex_replace(text_heard, RE_WS_TRAILING, "");
  
  outText = text_heard;

  // if (prob_n > 0) {
  //     prob /= prob_n;
  // }

  // const auto t_end = std::chrono::high_resolution_clock::now();
  // t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

  return true;
}

void WhisperSTT::shutdown() {
  whisper_free(ctx);
  ctx = nullptr;
}

// include/adapters/stt_whisper.h
#pragma once

#include "stt.h"
#include <string>
#include <vector>
#include "whisper.h"

/// Whisper-based STT adapter.
class WhisperSTT : public ISTT {
public:
    WhisperSTT() = default;

    /// Load the Whisper model from the given file.
    /// @param modelPath path to a ggml Whisper model.
    /// @return true on success, false on failure.
    bool init(const std::string &modelPath) override;

    /// Transcribe a single audio buffer.
    /// @param pcmf32 input audio.
    /// @param out resulting UTF-8 text.
    /// @return true on success, false on failure.
    bool transcribe(const std::vector<float> &pcmf32, std::string &outText) override;

    /// Release any resources held by Whisper.
    void shutdown() override;

private:
    whisper_context *ctx = nullptr;
    static constexpr int32_t MAX_TOKENS = 32;
    static const int32_t N_THREADS;
};

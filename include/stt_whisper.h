// include/adapters/stt_whisper.h
#pragma once

#include "stt.h"
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common-sdl.h"
#include "whisper.h"

/// Whisper-based STT adapter.
class WhisperSTT : public ISTT {
public:
    WhisperSTT() = default;

    /// Initialize Whisper.
    bool init() override;

    bool start_streaming(ResultCallback callback) override;
    void stop_streaming() override;

    /// Release any resources held by Whisper.
    void shutdown() override;

private:
    whisper_context *ctx = nullptr;
    std::unique_ptr<audio_async> audio_;
    ResultCallback callback_;
    std::thread streaming_thread_;
    std::atomic<bool> streaming_{false};
    std::atomic<bool> stop_streaming_{false};

    int sample_rate_ = 16000;
    int buffer_ms_ = 30000;
    float vad_threshold_ = 0.6f;
    int vad_capture_ms_ = 10000;

    static constexpr int vad_pre_window_ms_ = 2000;
    static constexpr int vad_start_ms_ = 1250;

    static constexpr int32_t MAX_TOKENS = 32;
    static const int32_t N_THREADS;

    bool init_audio();
    bool transcribe_buffer(const std::vector<float>& pcmf32, std::string &outText);
    void streaming_loop();
};

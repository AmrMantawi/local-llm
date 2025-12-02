// include/stt_sherpa.h
#pragma once

#include "stt.h"
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <functional>

// Sherpa-ONNX C++ API and ALSA helper
#include "sherpa-onnx/c-api/cxx-api.h"
#include "sherpa-onnx/csrc/alsa.h"

/// Sherpa-ONNX based STT adapter with integrated ALSA audio capture and VAD.
class SherpaSTT : public ISTT {
public:
    /// Callback function for transcription results
    /// Called when speech is detected and transcribed
    using TranscriptionCallback = std::function<void(const std::string&)>;

    SherpaSTT() = default;
    ~SherpaSTT() = default;

    /// Initialize Sherpa-ONNX backend.
    /// Model and device configuration are retrieved internally.
    /// @return true on success, false on failure
    bool init() override;

    /// Start continuous recognition with an internal audio/VAD loop.
    /// Returns true if streaming started successfully.
    bool start_streaming(TranscriptionCallback callback) override;

    /// Stop a previously started streaming loop.
    void stop_streaming() override;

    /// Release any resources held by Sherpa-ONNX.
    void shutdown() override;

private:
    // Runtime components
    std::unique_ptr<sherpa_onnx::cxx::OnlineRecognizer> recognizer_;
    std::unique_ptr<sherpa_onnx::cxx::VoiceActivityDetector> vad_;
    std::unique_ptr<sherpa_onnx::Alsa> alsa_;

    // Streaming state
    TranscriptionCallback callback_;
    std::thread streaming_thread_;
    std::atomic<bool> streaming_{false};
    std::atomic<bool> stop_streaming_{false};

    int sample_rate_ = 16000;
    int window_size_ = 512;

    bool initVad();
    void streaming_loop();
};


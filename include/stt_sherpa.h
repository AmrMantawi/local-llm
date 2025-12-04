// include/stt_sherpa.h
#pragma once

#include "stt.h"
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>

// Sherpa-ONNX C++ API and PortAudio microphone helper
#include "sherpa-onnx/c-api/cxx-api.h"
#include "sherpa-onnx/csrc/microphone.h"

/// Sherpa-ONNX based STT adapter with integrated microphone audio capture and VAD.
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
    std::unique_ptr<sherpa_onnx::Microphone> mic_;

    // Audio capture state (PortAudio callback → internal queue)
    std::mutex audio_mutex_;
    std::condition_variable audio_cv_;
    std::queue<std::vector<float>> audio_queue_;

    // Streaming state
    TranscriptionCallback callback_;
    std::thread streaming_thread_;
    std::atomic<bool> streaming_{false};
    std::atomic<bool> stop_streaming_{false};

    // Audio sample rates:
    // - mic_sample_rate_: actual PortAudio/device sampling rate
    // - model_sample_rate_: sampling rate expected by sherpa-onnx models/VAD
    int mic_sample_rate_ = 16000;
    int model_sample_rate_ = 16000;

    // Optional resampler used when mic_sample_rate_ != model_sample_rate_
    sherpa_onnx::cxx::LinearResampler resampler_;

    int window_size_ = 512;

    void streaming_loop();

    // PortAudio microphone callback (static member, implemented in .cpp)
    static int PortAudioCallback(const void *input_buffer,
                                 void *output_buffer,
                                 unsigned long frames_per_buffer,
                                 const PaStreamCallbackTimeInfo *time_info,
                                 PaStreamCallbackFlags status_flags,
                                 void *user_data);
};


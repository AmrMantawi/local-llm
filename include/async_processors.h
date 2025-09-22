#pragma once

#include "async_pipeline.h"
#include "stt.h"
#include "llm.h"
#include "tts.h"
#include "common-sdl.h"
#include "common.h"
#include "config_manager.h"
#include <sys/types.h>
#include <unistd.h>
#include <alsa/asoundlib.h>

namespace async_pipeline {

/**
 * STT processor that captures audio directly and produces transcribed text
 */
class STTProcessor : public BaseProcessor {
public:
    STTProcessor(SafeQueue<TextMessage>& output_queue, std::unique_ptr<ISTT> stt_backend);

protected:
    bool initialize() override;
    void process() override;
    void cleanup() override;
    bool handle_control_message(const ControlMessage& msg) override;

private:
    SafeQueue<TextMessage>& output_queue_;
    std::unique_ptr<ISTT> stt_;
    audio_async* audio_;
    
    // Cached config values
    int sample_rate_;
    int buffer_ms_;
    float vad_threshold_;
    int vad_capture_ms_;
    
    static const int vad_pre_window_ms_ = 2000;
    static const int vad_post_window_ms_ = 2000;
    
    bool is_in_speech_sequence_ = false;
    std::chrono::steady_clock::time_point speech_start_time_;
    std::vector<float> buffered_audio_;
};

/**
 * LLM processor that consumes text messages and generates responses
 */
class LLMProcessor : public BaseProcessor {
public:
    LLMProcessor(SafeQueue<TextMessage>& input_queue, SafeQueue<TextMessage>& output_queue,
                 std::unique_ptr<ILLM> llm_backend,
                 SafeQueue<TextMessage>* alt_input_queue = nullptr,
                 SafeQueue<TextMessage>* alt_output_queue = nullptr);

protected:
    bool initialize() override;
    void process() override;
    void cleanup() override;
    bool handle_control_message(const ControlMessage& msg) override;

private:
    SafeQueue<TextMessage>& input_queue_;
    SafeQueue<TextMessage>& output_queue_;
    SafeQueue<TextMessage>* alt_input_queue_;
    SafeQueue<TextMessage>* alt_output_queue_;
    std::unique_ptr<ILLM> llm_;
};

/**
 * Audio output processor that consumes audio chunks and plays them through ALSA
 */
class AudioOutputProcessor : public BaseProcessor {
public:
    AudioOutputProcessor(SafeQueue<AudioChunkMessage>& input_queue);
    
    // Immediate audio interruption - stops ALSA playback instantly
    void interrupt_audio_immediately();

protected:
    bool initialize() override;
    void process() override;
    void cleanup() override;
    bool handle_control_message(const ControlMessage& msg) override;

private:
    SafeQueue<AudioChunkMessage>& input_queue_;
    snd_pcm_t* alsa_handle_;
    unsigned int sample_rate_;
    
    bool init_audio_device();
    void close_audio_device();
    void play_audio_chunk(const std::vector<int16_t>& chunk);
};

/**
 * TTS processor that consumes responses and speaks them
 */
class TTSProcessor : public BaseProcessor {
public:
    TTSProcessor(SafeQueue<TextMessage>& input_queue, std::unique_ptr<ITTS> tts_backend, std::atomic<bool>* interrupt_flag = nullptr);

    void stop() override;

protected:
    bool initialize() override;
    void process() override;
    void cleanup() override;
    bool handle_control_message(const ControlMessage& msg) override;

private:
    SafeQueue<TextMessage>& input_queue_;
    std::unique_ptr<ITTS> tts_;
    std::atomic<bool> is_speaking_;
    pid_t tts_pid_;
    std::atomic<bool>* interrupt_flag_ = nullptr;
    
    // Internal audio output processing (not exposed externally)
    std::unique_ptr<SafeQueue<AudioChunkMessage>> audio_output_queue_;
    std::unique_ptr<AudioOutputProcessor> audio_output_processor_;
    
    void interrupt_current_speech();
};

} // namespace async_pipeline

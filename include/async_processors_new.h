#pragma once

#include "async_pipeline.h"
#include "stt.h"
#include "llm.h"
#include "tts.h"
#include "common-sdl.h"
#include "common.h"
#include "config_manager.h"
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>
#include <alsa/asoundlib.h>

namespace async_pipeline {

// Forward declarations
class AudioOutputProcessor;

/**
 * STT processor that captures audio directly and produces transcribed text
 */
class STTProcessor : public BaseProcessor {
public:
    STTProcessor(SafeQueue<TextMessage>& output_queue,
                 SafeQueue<ControlMessage>& control_queue, std::unique_ptr<ISTT> stt_backend);

protected:
    bool initialize() override;
    void process() override;
    void cleanup() override;

private:
    SafeQueue<TextMessage>& output_queue_;
    SafeQueue<ControlMessage>& control_queue_;
    std::unique_ptr<ISTT> stt_;
    audio_async* audio_;
    
    static const int vad_pre_window_ms_ = 1000;
    static const int vad_post_window_ms_ = 2000;
    static const float vad_threshold_;
    
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
                 SafeQueue<ControlMessage>& control_queue, std::unique_ptr<ILLM> llm_backend);

protected:
    bool initialize() override;
    void process() override;
    void cleanup() override;

private:
    SafeQueue<TextMessage>& input_queue_;
    SafeQueue<TextMessage>& output_queue_;
    SafeQueue<ControlMessage>& control_queue_;
    std::unique_ptr<ILLM> llm_;
};

/**
 * TTS processor that consumes responses and speaks them
 */
class TTSProcessor : public BaseProcessor {
public:
    TTSProcessor(SafeQueue<TextMessage>& input_queue, SafeQueue<ControlMessage>& control_queue,
                 std::unique_ptr<ITTS> tts_backend);

    void stop() override;

protected:
    bool initialize() override;
    void process() override;
    void cleanup() override;

private:
    SafeQueue<TextMessage>& input_queue_;
    SafeQueue<ControlMessage>& control_queue_;
    std::unique_ptr<ITTS> tts_;
    std::atomic<bool> is_speaking_;
    pid_t tts_pid_;
    
    // Internal audio output processing (not exposed externally)
    std::unique_ptr<SafeQueue<AudioChunkMessage>> audio_output_queue_;
    std::unique_ptr<AudioOutputProcessor> audio_output_processor_;
    
    void interrupt_current_speech();
};

/**
 * Audio output processor that consumes audio chunks and plays them through ALSA
 */
class AudioOutputProcessor : public BaseProcessor {
public:
    AudioOutputProcessor(SafeQueue<AudioChunkMessage>& input_queue, SafeQueue<ControlMessage>& control_queue);
    
    // Immediate audio interruption - stops ALSA playback instantly
    void interrupt_audio_immediately();

protected:
    bool initialize() override;
    void process() override;
    void cleanup() override;

private:
    SafeQueue<AudioChunkMessage>& input_queue_;
    SafeQueue<ControlMessage>& control_queue_;
    snd_pcm_t* alsa_handle_;
    unsigned int sample_rate_;
    
    bool init_audio_device();
    void close_audio_device();
    void play_audio_chunk(const std::vector<int16_t>& chunk);
};

} // namespace async_pipeline

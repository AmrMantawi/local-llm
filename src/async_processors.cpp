#include "async_processors.h"
#include <iostream>

namespace async_pipeline {

// STTProcessor implementation
STTProcessor::STTProcessor(SafeQueue<TextMessage>& output_queue,
                           SafeQueue<ControlMessage>& control_queue, std::unique_ptr<ISTT> stt_backend)
    : BaseProcessor("STTProcessor"), output_queue_(output_queue),
      control_queue_(control_queue), stt_(std::move(stt_backend)), audio_(nullptr),
      is_in_speech_sequence_(false) {
}

bool STTProcessor::initialize() {
    if (!stt_) {
        std::cerr << "[STTProcessor] No STT backend provided" << std::endl;
        return false;
    }
    
    auto& config = ConfigManager::getInstance();
    const std::string whisper_model_path = config.getSTTModelPath();
    
    if (!stt_->init(whisper_model_path)) {
        std::cerr << "[STTProcessor] Failed to initialize STT backend" << std::endl;
        return false;
    }
    
    // Cache config values once during initialization
    sample_rate_ = config.getAudioSampleRate();
    buffer_ms_ = config.getAudioBufferMs();
    vad_threshold_ = config.getVadThreshold();
    vad_capture_ms_ = config.getVadCaptureMs();
    
    // Initialize audio capture
    audio_ = new audio_async(buffer_ms_);
    bool audio_initialized = false;
    for (int attempt = 1; attempt <= 8; ++attempt) {
        std::cout << "[STTProcessor] Audio init attempt " << attempt << "/8..." << std::endl;
        if (audio_->init(-1, sample_rate_)) {
            audio_initialized = true;
            std::cout << "[STTProcessor] Audio initialization successful on attempt " << attempt << std::endl;
            break;
        }
        std::cerr << "[STTProcessor] Audio init attempt " << attempt << " failed, retrying in 500ms..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    if (!audio_initialized) {
        std::cerr << "[STTProcessor] Failed to initialize audio capture after 8 attempts" << std::endl;
        return false;
    }
    
    if (!audio_->resume()) {
        std::cerr << "[STTProcessor] Failed to start audio capture" << std::endl;
        return false;
    }
    
    std::cout << "[STTProcessor] Initialized successfully" << std::endl;
    return true;
}

void STTProcessor::process() {
    // Check if we should stop processing
    if (!is_running()) {
        return;
    }
    
    // Check for control messages first
    ControlMessage control_msg(ControlMessage::INTERRUPT);
    if (control_queue_.try_pop(control_msg)) {
        if (control_msg.type == ControlMessage::INTERRUPT || 
            control_msg.type == ControlMessage::FLUSH_QUEUES) {
            // Flush our output queue
            size_t flushed = output_queue_.flush();
            if (flushed > 0) {
                std::cout << "[STTProcessor] Flushed " << flushed << " pending text messages" << std::endl;
            }
            // Forward interrupt signal to downstream processors
            control_queue_.push(std::move(control_msg), std::chrono::milliseconds(10));
            return;
        }
    }
    
    if (!audio_) {
        std::cerr << "[STTProcessor] WARNING: audio_ is null!" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return;
    }
    
    std::vector<float> audio;
    
    // Sleep to avoid busy loop - this is needed for continuous audio monitoring
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Get recent audio for VAD analysis
    audio_->get(vad_pre_window_ms_, audio);
    
    if (audio.empty()) {
        return;
    }
    
    // Use cached config values (no repeated config access!)
    bool voice_detected = ::vad_simple(
        audio, 
        sample_rate_, 
        1250, // vad_start_ms
        vad_threshold_, 
        100.0f, // vad_freq_cutoff
        false // vad_print_energy
    );
    
    if (voice_detected) {
        // Capture full audio window
        audio_->get(vad_capture_ms_, audio);
        
        if (!audio.empty()) {
            // Transcribe audio directly
            std::string transcribed_text;
            bool success = stt_->transcribe(audio, transcribed_text);
            
            if (success && !transcribed_text.empty()) {
                // Create text message
                TextMessage text_msg(transcribed_text);
                
                // Push to output queue
                if (!output_queue_.push(std::move(text_msg), std::chrono::milliseconds(500))) {
                    std::cerr << "[STTProcessor] Failed to push text message (queue full)" << std::endl;
                } else {
                    std::cout << "[STTProcessor] → " << transcribed_text << std::endl;
                }
            }
            
            // Clear audio buffer
            audio_->clear();
        }
    }
}

void STTProcessor::cleanup() {
    if (audio_) {
        audio_->pause();
        delete audio_;
        audio_ = nullptr;
    }
    if (stt_) {
        stt_->shutdown();
    }
    std::cout << "[STTProcessor] Cleanup completed" << std::endl;
}

// LLMProcessor implementation
LLMProcessor::LLMProcessor(SafeQueue<TextMessage>& input_queue, SafeQueue<TextMessage>& output_queue,
                           SafeQueue<ControlMessage>& control_queue, std::unique_ptr<ILLM> llm_backend)
    : BaseProcessor("LLMProcessor"), input_queue_(input_queue), output_queue_(output_queue),
      control_queue_(control_queue), llm_(std::move(llm_backend)) {
}

bool LLMProcessor::initialize() {
    if (!llm_) {
        std::cerr << "[LLMProcessor] No LLM backend provided" << std::endl;
        return false;
    }
    
    // Initialize LLM backend (consistent with TTSProcessor pattern)
    auto& config = ConfigManager::getInstance();
    const std::string llama_model_path = config.getLLMModelPath();
    
    if (!llm_->init(llama_model_path)) {
        std::cerr << "[LLMProcessor] Failed to initialize LLM backend" << std::endl;
        return false;
    }
    
    std::cout << "[LLMProcessor] Initialized successfully" << std::endl;
    return true;
}

void LLMProcessor::process() {
    // Check if we should stop processing
    if (!is_running()) {
        return;
    }
    
    // Check for control messages first
    ControlMessage control_msg(ControlMessage::INTERRUPT);
    if (control_queue_.try_pop(control_msg)) {
        if (control_msg.type == ControlMessage::INTERRUPT || 
            control_msg.type == ControlMessage::FLUSH_QUEUES) {
            // Flush input and output queues
            size_t input_flushed = input_queue_.flush();
            size_t output_flushed = output_queue_.flush();
            if (input_flushed > 0 || output_flushed > 0) {
                std::cout << "[LLMProcessor] Flushed " << input_flushed << " input and " 
                          << output_flushed << " output messages" << std::endl;
            }
            // Forward interrupt signal to downstream processors
            control_queue_.push(std::move(control_msg), std::chrono::milliseconds(10));
            return;
        }
    }
    
    TextMessage input_msg;
    if (input_queue_.try_pop(input_msg)) {
        std::cout << "[LLMProcessor] Processing: " << input_msg.text << std::endl;
        
        // Generate response
        std::string response;
        bool success = llm_->generate_async(input_msg.text, response, 
            [this](const std::string& text_chunk) {
                // Create response message
                TextMessage response_msg(text_chunk);
                
                // Push to output queue
                if (!output_queue_.push(std::move(response_msg), std::chrono::milliseconds(1000))) {
                    std::cerr << "[LLMProcessor] Failed to push response message (queue full)" << std::endl;
                } else {
                    std::cout << "[LLMProcessor] → " << text_chunk << std::endl;
                }
            });
        
        if (!success) {
            std::cerr << "[LLMProcessor] Failed to generate response for: " << input_msg.text << std::endl;
        }
    } else {
        // No input available, sleep briefly
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void LLMProcessor::cleanup() {
    if (llm_) {
        llm_->shutdown();
    }
    std::cout << "[LLMProcessor] Cleanup completed" << std::endl;
}

// TTSProcessor implementation
TTSProcessor::TTSProcessor(SafeQueue<TextMessage>& input_queue, SafeQueue<ControlMessage>& control_queue,
                           std::unique_ptr<ITTS> tts_backend)
    : BaseProcessor("TTSProcessor"), input_queue_(input_queue), control_queue_(control_queue),
      tts_(std::move(tts_backend)), is_speaking_(false), tts_pid_(-1) {
}

void TTSProcessor::stop() {
    if (!is_running()) return;
    
    // Now call the base stop() method to join the thread
    BaseProcessor::stop();
}

bool TTSProcessor::initialize() {
    if (!tts_) {
        std::cerr << "[TTSProcessor] No TTS backend provided" << std::endl;
        return false;
    }
    
    // Create audio output queue first
    audio_output_queue_ = std::make_unique<SafeQueue<AudioChunkMessage>>(50); // 50 audio chunks max
    
    if (!tts_->init()) {
        std::cerr << "[TTSProcessor] Failed to initialize TTS backend" << std::endl;
        return false;
    }
            
    // Initialize AudioOutputProcessor with the queue
    audio_output_processor_ = std::make_unique<AudioOutputProcessor>(*audio_output_queue_, control_queue_);
    if (!audio_output_processor_->start()) {
        std::cerr << "[TTSProcessor] Failed to start AudioOutputProcessor" << std::endl;
        return false;
    }
    
    std::cout << "[TTSProcessor] Initialized successfully with audio output processor" << std::endl;
    return true;
}

void TTSProcessor::process() {
    // Check if we should stop processing
    if (!is_running()) {
        return;
    }
    
    // Check for control messages first
    ControlMessage control_msg(ControlMessage::INTERRUPT);
    if (control_queue_.try_pop(control_msg)) {
        if (control_msg.type == ControlMessage::INTERRUPT || 
            control_msg.type == ControlMessage::FLUSH_QUEUES) {
            // Flush input queue
            size_t flushed = input_queue_.flush();
            if (flushed > 0) {
                std::cout << "[TTSProcessor] Interrupted! Flushed " << flushed << " pending TTS messages" << std::endl;
            }
            // Stop current speech gracefully
            interrupt_current_speech();
            // Clear interrupt flag after handling interruption
            clear_interrupt();
            std::cout << "[TTSProcessor] Interrupt handled, ready for new speech" << std::endl;
            return;
        }
    }
    
    TextMessage text_msg;
    if (!input_queue_.try_pop(text_msg)) {
        // No input available, sleep briefly
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return;
    }
    
    // Check for interruption before speaking (but don't skip if we just cleared it)
    if (is_interrupt_requested()) {
        std::cout << "[TTSProcessor] Clearing stale interrupt flag before speaking" << std::endl;
        clear_interrupt();
    }
    
    // Speak the chunk and get audio data
    std::cout << "[TTSProcessor] Speaking: " << text_msg.text << std::endl;
    
    // Create AudioChunkMessage to receive the audio data
    AudioChunkMessage audio_chunk;
    bool success = tts_->speak(text_msg.text, audio_chunk);
    
    if (success && !audio_chunk.audio_data.empty()) {
        // Queue the audio chunk for playback
        if (!audio_output_queue_->push(std::move(audio_chunk), std::chrono::milliseconds(1000))) {
            std::cerr << "[TTSProcessor] Failed to queue audio chunk (queue full)" << std::endl;
        } else {
            std::cout << "[TTSProcessor] Queued audio chunk with " << audio_chunk.audio_data.size() 
                      << " samples at " << audio_chunk.sample_rate << " Hz" << std::endl;
        }
    } else {
        std::cerr << "[TTSProcessor] Failed to speak: " << text_msg.text << std::endl;
    }
}

void TTSProcessor::cleanup() {
    // Stop internal audio output processor first
    if (audio_output_processor_) {
        audio_output_processor_->stop();
        audio_output_processor_.reset();
    }
    
    if (tts_) {
        tts_->shutdown();
    }
    std::cout << "[TTSProcessor] Cleanup completed" << std::endl;
}

void TTSProcessor::interrupt_current_speech() {
    if (audio_output_processor_) {
        std::cout << "[TTSProcessor] Using immediate audio interruption" << std::endl;
        audio_output_processor_->interrupt_audio_immediately();
    }
    
    // TTS backends no longer support interrupt since they generate audio synchronously
}

// AudioOutputProcessor implementation
AudioOutputProcessor::AudioOutputProcessor(SafeQueue<AudioChunkMessage>& input_queue, SafeQueue<ControlMessage>& control_queue)
    : BaseProcessor("AudioOutputProcessor"), input_queue_(input_queue), control_queue_(control_queue),
      alsa_handle_(nullptr), sample_rate_(22050) {
}

void AudioOutputProcessor::interrupt_audio_immediately() {
    // Clear the audio queue to prevent queued audio from playing
    size_t flushed = input_queue_.flush();
    if (flushed > 0) {
        std::cout << "[AudioOutputProcessor] Flushed " << flushed << " queued audio chunks" << std::endl;
    }
    
    // Stop ALSA playback immediately
    if (alsa_handle_) {
        snd_pcm_drop(alsa_handle_);  // Stop immediately, don't drain buffer
        snd_pcm_prepare(alsa_handle_); // Prepare for next playback
        std::cout << "[AudioOutputProcessor] Stopped ALSA playback immediately" << std::endl;
    }
}

bool AudioOutputProcessor::initialize() {
    std::cout << "[AudioOutputProcessor] Initializing ALSA..." << std::endl;
    return init_audio_device();
}

void AudioOutputProcessor::process() {
    // Check if we should stop processing
    if (!is_running()) {
        return;
    }
    
    // Check for control messages (though we don't handle them directly)
    ControlMessage control_msg(ControlMessage::INTERRUPT);
    if (control_queue_.try_pop(control_msg)) {
        // Forward control messages to other processors
        control_queue_.push(std::move(control_msg), std::chrono::milliseconds(10));
    }
    
    AudioChunkMessage audio_msg;
    if (input_queue_.try_pop(audio_msg)) {
        if (!audio_msg.audio_data.empty()) {
            // Update sample rate if different
            if (audio_msg.sample_rate != sample_rate_) {
                sample_rate_ = audio_msg.sample_rate;
                // Reinitialize ALSA with new sample rate
                close_audio_device();
                init_audio_device();
            }
            
            play_audio_chunk(audio_msg.audio_data);
        }
    } else {
        // No input available, sleep briefly
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void AudioOutputProcessor::cleanup() {
    close_audio_device();
    std::cout << "[AudioOutputProcessor] Cleanup completed" << std::endl;
}

bool AudioOutputProcessor::init_audio_device() {
    int err;
    
    if ((err = snd_pcm_open(&alsa_handle_, "default", SND_PCM_STREAM_PLAYBACK, 0)) < 0) {
        std::cerr << "[AudioOutputProcessor] Cannot open audio device: " << snd_strerror(err) << std::endl;
        return false;
    }
    
    // Set hardware parameters
    snd_pcm_hw_params_t *params;
    snd_pcm_hw_params_alloca(&params);
    snd_pcm_hw_params_any(alsa_handle_, params);
    snd_pcm_hw_params_set_access(alsa_handle_, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(alsa_handle_, params, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(alsa_handle_, params, 1);
    
    // Set sample rate
    unsigned int rate = sample_rate_;
    if ((err = snd_pcm_hw_params_set_rate_near(alsa_handle_, params, &rate, 0)) < 0) {
        std::cerr << "[AudioOutputProcessor] Cannot set sample rate: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_);
        alsa_handle_ = nullptr;
        return false;
    }
    
    // Set period size for low latency
    snd_pcm_uframes_t period_size = 1024;
    if ((err = snd_pcm_hw_params_set_period_size_near(alsa_handle_, params, &period_size, 0)) < 0) {
        std::cerr << "[AudioOutputProcessor] Cannot set period size: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_);
        alsa_handle_ = nullptr;
        return false;
    }
    
    // Set buffer size
    snd_pcm_uframes_t buffer_size = period_size * 4;
    if ((err = snd_pcm_hw_params_set_buffer_size_near(alsa_handle_, params, &buffer_size)) < 0) {
        std::cerr << "[AudioOutputProcessor] Cannot set buffer size: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_);
        alsa_handle_ = nullptr;
        return false;
    }
    
    if ((err = snd_pcm_hw_params(alsa_handle_, params)) < 0) {
        std::cerr << "[AudioOutputProcessor] Cannot set parameters: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_);
        alsa_handle_ = nullptr;
        return false;
    }
    
    // Prepare the PCM device
    if ((err = snd_pcm_prepare(alsa_handle_)) < 0) {
        std::cerr << "[AudioOutputProcessor] Cannot prepare audio device: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_);
        alsa_handle_ = nullptr;
        return false;
    }
    
    std::cout << "[AudioOutputProcessor] ALSA initialized successfully at " << rate << " Hz" << std::endl;
    return true;
}

void AudioOutputProcessor::close_audio_device() {
    if (alsa_handle_) {
        snd_pcm_drain(alsa_handle_);
        snd_pcm_close(alsa_handle_);
        alsa_handle_ = nullptr;
    }
}

void AudioOutputProcessor::play_audio_chunk(const std::vector<int16_t>& chunk) {
    if (chunk.empty() || !alsa_handle_) {
        return;
    }
    
    snd_pcm_sframes_t frames = snd_pcm_writei(alsa_handle_, chunk.data(), chunk.size());
    if (frames < 0) {
        // Handle underrun
        if (frames == -EPIPE) {
            std::cerr << "[AudioOutputProcessor] ALSA underrun, recovering..." << std::endl;
            snd_pcm_prepare(alsa_handle_);
        } else {
            std::cerr << "[AudioOutputProcessor] ALSA write error: " << snd_strerror(frames) << std::endl;
        }
    }
}

} // namespace async_pipeline

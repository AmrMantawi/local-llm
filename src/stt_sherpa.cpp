#include "stt_sherpa.h"

#include "config_manager.h"

#include <algorithm>
#include <iostream>
#include <thread>

using sherpa_onnx::cxx::OnlineRecognizer;
using sherpa_onnx::cxx::OnlineRecognizerConfig;
using sherpa_onnx::cxx::OnlineStream;
using sherpa_onnx::cxx::LinearResampler;
using sherpa_onnx::cxx::VadModelConfig;
using sherpa_onnx::cxx::VoiceActivityDetector;

// PortAudio microphone callback: push audio frames into the owning SherpaSTT
// instance's internal queue.
int SherpaSTT::PortAudioCallback(const void *input_buffer,
                                 void * /*output_buffer*/,
                                 unsigned long frames_per_buffer,
                                 const PaStreamCallbackTimeInfo * /*time_info*/,
                                 PaStreamCallbackFlags /*status_flags*/,
                                 void *user_data) {
    auto *self = static_cast<SherpaSTT *>(user_data);
    if (!self || !input_buffer || frames_per_buffer == 0) {
        return paContinue;
    }

    const float *in = static_cast<const float *>(input_buffer);
    std::vector<float> chunk(in, in + frames_per_buffer);

    {
        std::lock_guard<std::mutex> lock(self->audio_mutex_);
        self->audio_queue_.emplace(std::move(chunk));
    }
    self->audio_cv_.notify_one();

    return paContinue;
}

namespace {

// Helper to create a sherpa VAD instance from a single model path.
VoiceActivityDetector CreateVad(const std::string &modelPath, int32_t sample_rate) {
    VadModelConfig config;
    config.silero_vad.model = modelPath;
    config.silero_vad.threshold = 0.3f;
    config.silero_vad.min_silence_duration = 0.25f;
    config.silero_vad.min_speech_duration = 0.01f;
    config.silero_vad.window_size = 512;
    config.silero_vad.max_speech_duration = 8.0f;

    config.sample_rate = sample_rate;
    config.num_threads = 1;
    config.provider = "cpu";
    config.debug = false;

    auto vad = VoiceActivityDetector::Create(config, /*buffer_size_in_seconds=*/60.0f);
    if (!vad.Get()) {
        std::cerr << "[SherpaSTT] Failed to create VAD (silero_vad model: " << modelPath
                  << ")" << std::endl;
    }

    return vad;
}

OnlineRecognizer CreateOnlineRecognizer(
    const std::string &encoderPath,
    const std::string &decoderPath,
    const std::string &joinerPath,
    const std::string &tokensPath,
    int32_t numThreads,
    int32_t sampleRate) {
    OnlineRecognizerConfig config;

    config.model_config.transducer.encoder = encoderPath;
    config.model_config.transducer.decoder = decoderPath;
    config.model_config.transducer.joiner = joinerPath;
    config.model_config.tokens = tokensPath;

    config.model_config.num_threads = numThreads;
    config.model_config.provider = "cpu";
    config.model_config.debug = false;

    config.feat_config.sample_rate = sampleRate;
    config.feat_config.feature_dim = 80;

    config.decoding_method = "greedy_search";

    std::cout << "[SherpaSTT] Loading sherpa-onnx model..." << std::endl;
    OnlineRecognizer recognizer = OnlineRecognizer::Create(config);
    if (!recognizer.Get()) {
        std::cerr << "[SherpaSTT] Failed to create OnlineRecognizer" << std::endl;
    } else {
        std::cout << "[SherpaSTT] Model loaded." << std::endl;
    }

    return recognizer;
}

} // namespace

bool SherpaSTT::init() {
    auto &config = ConfigManager::getInstance();

    const std::string vadPath =
        config.getNestedModelPath("stt", "sherpa", "vad");
    const std::string encoderPath =
        config.getNestedModelPath("stt", "sherpa", "encoder");
    const std::string decoderPath =
        config.getNestedModelPath("stt", "sherpa", "decoder");
    const std::string joinerPath =
        config.getNestedModelPath("stt", "sherpa", "joiner");
    const std::string tokensPath =
        config.getNestedModelPath("stt", "sherpa", "tokens");

    mic_sample_rate_ = config.getAudioSampleRate();
    if (mic_sample_rate_ <= 0) {
        mic_sample_rate_ = 16000;
    }

    // sherpa-onnx Zipformer models and Silero VAD expect 16 kHz audio.
    model_sample_rate_ = 16000;

    if (vadPath.empty()) {
        std::cerr << "[SherpaSTT] Sherpa VAD model not found" << std::endl;
        return false;
    }
    if (encoderPath.empty() || decoderPath.empty() || joinerPath.empty()) {
        std::cerr << "[SherpaSTT] Sherpa transducer model paths (encoder/decoder/joiner) "
                  << "are not fully configured" << std::endl;
        return false;
    }
    if (tokensPath.empty()) {
        std::cerr << "[SherpaSTT] Sherpa tokens file not found" << std::endl;
        return false;
    }

    const int32_t numThreads =
        std::max<int32_t>(1, std::min<int32_t>(4, std::thread::hardware_concurrency()));

    auto recognizer = CreateOnlineRecognizer(
        encoderPath, decoderPath, joinerPath, tokensPath, numThreads,
        model_sample_rate_);
    if (!recognizer.Get()) {
        return false;
    }

    recognizer_ = std::make_unique<OnlineRecognizer>(std::move(recognizer));

    auto vad = CreateVad(vadPath, model_sample_rate_);
    if (!vad.Get()) {
        return false;
    }

    vad_ = std::make_unique<VoiceActivityDetector>(std::move(vad));

    // Initialize PortAudio microphone helper; actual device is opened in
    // start_streaming() so that we don't capture audio until needed.
    mic_ = std::make_unique<sherpa_onnx::Microphone>();

    // Configure optional resampler if microphone rate differs from model rate.
    if (mic_sample_rate_ != model_sample_rate_) {
        float min_freq =
            std::min(static_cast<float>(mic_sample_rate_),
                     static_cast<float>(model_sample_rate_));
        float lowpass_cutoff = 0.99f * 0.5f * min_freq;

        int32_t lowpass_filter_width = 6;
        resampler_ = LinearResampler::Create(
            mic_sample_rate_, model_sample_rate_,
            lowpass_cutoff, lowpass_filter_width);

        std::cout << "[SherpaSTT] Using LinearResampler from "
                  << mic_sample_rate_ << " Hz to "
                  << model_sample_rate_ << " Hz" << std::endl;
    } else {
        std::cout << "[SherpaSTT] Mic sample rate matches model ("
                  << model_sample_rate_ << " Hz); no resampler needed"
                  << std::endl;
    }

    window_size_ = 512;

    std::cout << "[SherpaSTT] Initialized successfully" << std::endl;
    return true;
}

bool SherpaSTT::start_streaming(TranscriptionCallback callback) {
    if (!callback) {
        std::cerr << "[SherpaSTT] Streaming callback is empty" << std::endl;
        return false;
    }

    if (!recognizer_ || !vad_ || !mic_) {
        std::cerr << "[SherpaSTT] Cannot start streaming: not initialized" << std::endl;
        return false;
    }

    if (streaming_) {
        std::cerr << "[SherpaSTT] Streaming already in progress" << std::endl;
        return false;
    }

    callback_ = std::move(callback);

    // Open default microphone device with PortAudio. Device selection can be
    // customized later via environment variables (similar to sherpa-onnx).
    int32_t device_index = mic_->GetDefaultInputDevice();
    if (device_index < 0) {
        std::cerr << "[SherpaSTT] No default input device found (PortAudio)" << std::endl;
        return false;
    }

    mic_->PrintDevices(device_index);

    if (!mic_->OpenDevice(device_index, mic_sample_rate_, 1, PortAudioCallback, this)) {
        std::cerr << "[SherpaSTT] Failed to open PortAudio microphone device index "
                  << device_index << std::endl;
        return false;
    }
    stop_streaming_ = false;
    streaming_thread_ = std::thread(&SherpaSTT::streaming_loop, this);
    streaming_ = true;

    return true;
}

void SherpaSTT::stop_streaming() {
    if (!streaming_) {
        return;
    }

    stop_streaming_ = true;
    if (streaming_thread_.joinable()) {
        streaming_thread_.join();
    }

    streaming_ = false;
    stop_streaming_ = false;
    callback_ = nullptr;
}

void SherpaSTT::shutdown() {
    stop_streaming();

    if (mic_) {
        mic_->CloseDevice();
        mic_.reset();
    }
    // recognizer_ and vad_ are RAII wrappers and will clean up in their destructors.
}

void SherpaSTT::streaming_loop() {
    bool speech_started = false;
    int32_t segment_id = 0;
    int32_t offset = 0;
    std::vector<float> buffer;

    // Tail paddings appended after each VAD speech segment before finalizing
    // the recognizer stream. This helps the model flush its internal state.
    const float tail_padding_len = 1.28f;  // seconds; tuned to model chunk size
    std::vector<float> tail_paddings(
        static_cast<int32_t>(tail_padding_len * model_sample_rate_), 0.0f);

    while (!stop_streaming_) {
        // Wait for audio from the PortAudio callback
        std::vector<float> chunk;
        {
            std::unique_lock<std::mutex> lock(audio_mutex_);
            audio_cv_.wait(lock, [this] {
                return stop_streaming_ || !audio_queue_.empty();
            });

            if (stop_streaming_) {
                break;
            }

            if (!audio_queue_.empty()) {
                chunk = std::move(audio_queue_.front());
                audio_queue_.pop();
            }
        }

        if (chunk.empty()) {
            continue;
        }

        // Resample microphone audio to the model/VAD sample rate if needed.
        if (!resampler_.Get()) {
            buffer.insert(buffer.end(), chunk.begin(), chunk.end());
        } else {
            auto resampled =
                resampler_.Resample(chunk.data(),
                                    static_cast<int32_t>(chunk.size()),
                                    /*flush=*/false);
            buffer.insert(buffer.end(), resampled.begin(), resampled.end());
        }

        // VAD processing
        for (; offset + window_size_ < static_cast<int32_t>(buffer.size());
             offset += window_size_) {
            vad_->AcceptWaveform(buffer.data() + offset, window_size_);
            if (vad_->IsDetected() && !speech_started) {
                speech_started = true;
                ++segment_id;
                std::cerr << "[SherpaSTT] VAD detected speech, segment "
                          << segment_id << std::endl;
            } else if (!vad_->IsDetected() && speech_started) {
                speech_started = false;
                std::cerr << "[SherpaSTT] VAD lost speech, segment "
                          << segment_id << " ended (pending flush)" << std::endl;
            }
        }

        // if (!speech_started) {
        //     if (buffer.size() > static_cast<size_t>(10 * window_size_)) {
        //         offset -= static_cast<int32_t>(buffer.size()) - 10 * window_size_;
        //         buffer = {buffer.end() - 10 * window_size_, buffer.end()};
        //     }
        // }

        while (!vad_->IsEmpty()) {
            auto segment = vad_->Front();
            auto speech = segment.samples;

            if (speech.empty()) {
                vad_->Pop();
                continue;
            }

            std::cerr << "[SherpaSTT] Processing VAD segment(" << segment_id
                      << ") with " << speech.size() << " samples" << std::endl;

            OnlineStream stream = recognizer_->CreateStream();
            stream.AcceptWaveform(
                model_sample_rate_, speech.data(),
                static_cast<int32_t>(speech.size()));
            stream.AcceptWaveform(
                model_sample_rate_, tail_paddings.data(),
                static_cast<int32_t>(tail_paddings.size()));
            stream.InputFinished();

            while (recognizer_->IsReady(&stream)) {
                recognizer_->Decode(&stream);
            }

            auto result = recognizer_->GetResult(&stream);
            std::cerr << "[SherpaSTT] Recognizer result for segment(" << segment_id
                      << "): '" << result.text << "'" << std::endl;

            if (!result.text.empty() && callback_) {
                callback_(result.text);
                std::cout << "[SherpaSTT] vad segment(" << segment_id << ") → "
                          << result.text << std::endl;
            }

            vad_->Pop();

            buffer.clear();
            offset = 0;
            speech_started = false;
        }
    }
}


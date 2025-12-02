#include "stt_sherpa.h"

#include "config_manager.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

using namespace std::chrono_literals;

using sherpa_onnx::Alsa;
using sherpa_onnx::cxx::OnlineRecognizer;
using sherpa_onnx::cxx::OnlineRecognizerConfig;
using sherpa_onnx::cxx::OnlineStream;
using sherpa_onnx::cxx::VadModelConfig;
using sherpa_onnx::cxx::VoiceActivityDetector;

namespace {

// Helper to create a sherpa VAD instance from a single model path.
VoiceActivityDetector CreateVad(const std::string &modelPath, int32_t sample_rate) {
    VadModelConfig config;
    config.silero_vad.model = modelPath;
    config.silero_vad.threshold = 0.5f;
    config.silero_vad.min_silence_duration = 0.25f;
    config.silero_vad.min_speech_duration = 0.25f;
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
    int32_t numThreads) {
    OnlineRecognizerConfig config;

    config.model_config.transducer.encoder = encoderPath;
    config.model_config.transducer.decoder = decoderPath;
    config.model_config.transducer.joiner = joinerPath;
    config.model_config.tokens = tokensPath;

    config.model_config.num_threads = numThreads;
    config.model_config.provider = "cpu";
    config.model_config.debug = false;

    config.feat_config.sample_rate = 16000;
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

    sample_rate_ = config.getAudioSampleRate();
    if (sample_rate_ <= 0) {
        sample_rate_ = 16000;
    }

    // Select ALSA device from config (falls back to "default" internally)
    const std::string audio_device = config.getAudioDevice();

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
        encoderPath, decoderPath, joinerPath, tokensPath, numThreads);
    if (!recognizer.Get()) {
        return false;
    }

    recognizer_ = std::make_unique<OnlineRecognizer>(std::move(recognizer));

    auto vad = CreateVad(vadPath, 16000);
    if (!vad.Get()) {
        return false;
    }

    vad_ = std::make_unique<VoiceActivityDetector>(std::move(vad));

    // ALSA always provides audio at its own native rate; internal resampling is
    // handled by the sherpa_onnx::Alsa class.
    try {
        alsa_ = std::make_unique<Alsa>(audio_device.c_str());
    } catch (const std::exception &e) {
        std::cerr << "[SherpaSTT] Failed to initialize ALSA device '" << audio_device
                  << "': " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "[SherpaSTT] Failed to initialize ALSA device '" << audio_device
                  << "' (unknown error)" << std::endl;
        return false;
    }

    window_size_ = 512; // in samples, matches VAD window_size above

    std::cout << "[SherpaSTT] Initialized successfully" << std::endl;
    return true;
}

bool SherpaSTT::start_streaming(TranscriptionCallback callback) {
    if (!callback) {
        std::cerr << "[SherpaSTT] Streaming callback is empty" << std::endl;
        return false;
    }

    if (!recognizer_ || !vad_ || !alsa_) {
        std::cerr << "[SherpaSTT] Cannot start streaming: not initialized" << std::endl;
        return false;
    }

    if (streaming_) {
        std::cerr << "[SherpaSTT] Streaming already in progress" << std::endl;
        return false;
    }

    callback_ = std::move(callback);
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

    alsa_.reset();
    // recognizer_ and vad_ are RAII wrappers and will clean up in their destructors.
}

bool SherpaSTT::initVad() {
    // init() already initializes VAD based on the config; provide a thin
    // wrapper so that future extensions can reconfigure VAD independently.
    auto &config = ConfigManager::getInstance();
    const std::string vadPath =
        config.getNestedModelPath("stt", "sherpa", "vad");

    if (vadPath.empty()) {
        std::cerr << "[SherpaSTT] initVad: VAD model path is empty" << std::endl;
        return false;
    }

    auto vad = CreateVad(vadPath, 16000);
    if (!vad.Get()) {
        return false;
    }

    vad_ = std::make_unique<VoiceActivityDetector>(std::move(vad));

    return true;
}

void SherpaSTT::streaming_loop() {
    bool speech_started = false;
    int32_t segment_id = 0;

    while (!stop_streaming_) {
        if (!alsa_) {
            break;
        }

        // This is a blocking read.
        const auto &samples = alsa_->Read(window_size_);
        if (samples.empty()) {
            std::this_thread::sleep_for(10ms);
            continue;
        }

        if (!vad_) {
            // If VAD is not available for some reason, do a simple "always on"
            // recognition on fixed windows.
            OnlineStream stream = recognizer_->CreateStream();
            stream.AcceptWaveform(16000, samples.data(),
                                  static_cast<int32_t>(samples.size()));
            stream.InputFinished();

            while (recognizer_->IsReady(&stream)) {
                recognizer_->Decode(&stream);
            }

            auto result = recognizer_->GetResult(&stream);
            if (!result.text.empty() && callback_) {
                callback_(result.text);
            }

            continue;
        }

        vad_->AcceptWaveform(samples.data(),
                             static_cast<int32_t>(samples.size()));

        if (vad_->IsDetected() && !speech_started) {
            speech_started = true;
            ++segment_id;
        } else if (!vad_->IsDetected() && speech_started) {
            speech_started = false;
        }

        while (!vad_->IsEmpty()) {
            auto segment = vad_->Front();
            auto speech = segment.samples;

            if (speech.empty()) {
                vad_->Pop();
                continue;
            }

            OnlineStream stream = recognizer_->CreateStream();
            stream.AcceptWaveform(16000, speech.data(),
                                  static_cast<int32_t>(speech.size()));
            stream.InputFinished();

            while (recognizer_->IsReady(&stream)) {
                recognizer_->Decode(&stream);
            }

            auto result = recognizer_->GetResult(&stream);
            if (!result.text.empty() && callback_) {
                callback_(result.text);
                std::cout << "[SherpaSTT] vad segment(" << segment_id << ") → "
                          << result.text << std::endl;
            }

            vad_->Pop();
        }
    }
}


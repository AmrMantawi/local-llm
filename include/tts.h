#pragma once
#include <string>
#include <vector>

// Forward declaration
namespace async_pipeline {
    struct AudioChunkMessage;
}

// Forward declaration for phoneme timing
struct PhonemeTimingInfo {
    int64_t phoneme_id;
    float duration_seconds;
};

class ITTS {
public:
  /// Initialize TTS.
  /// @return true on success, false on failure.
  virtual bool init() = 0;

  /// Speak the given text and write audio to AudioChunkMessage.
  /// @param text UTF-8 text to speak.
  /// @param audio_chunk AudioChunkMessage to write audio data to.
  /// @return true on success, false on failure.
  virtual bool speak(const std::string &text, async_pipeline::AudioChunkMessage& audio_chunk) = 0;

  /// Speak the given text with phoneme timing information.
  /// @param text UTF-8 text to speak.
  /// @param audio_chunk AudioChunkMessage to write audio data to.
  /// @param phoneme_timings Vector to store phoneme timing information.
  /// @return true on success, false on failure.
  virtual bool speakWithPhonemeTimings(const std::string &text, async_pipeline::AudioChunkMessage& audio_chunk, std::vector<PhonemeTimingInfo>& phoneme_timings) = 0;

  // TODO: Add async speak with callback for streaming audio

  /// Release any resources held by TTS.
  virtual void shutdown() = 0;
};


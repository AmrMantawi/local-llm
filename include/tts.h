#pragma once
#include <string>

class ITTS {
public:
  /// Initialize TTS.
  /// @return true on success, false on failure.
  virtual bool init() = 0;

  /// Speak the given text.
  /// @param text UTF-8 text to speak.
  /// @return true on success, false on failure.
  virtual bool speak(const std::string &text) = 0;

  /// Release any resources held by TTS.
  virtual void shutdown() = 0;
};


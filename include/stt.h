#pragma once

#include <functional>
#include <string>

class ISTT {
public:
    using ResultCallback = std::function<void(const std::string&)>;

    /// Initialize STT. Model path is retrieved internally.
    virtual bool init() = 0;

    /// Start continuous recognition with an internal audio/VAD loop.
    /// Returns true if streaming started successfully.
    virtual bool start_streaming(ResultCallback) { return false; }

    /// Stop a previously started streaming loop.
    virtual void stop_streaming() {}

    /// Release any resources held by STT.
    virtual void shutdown() = 0;
};


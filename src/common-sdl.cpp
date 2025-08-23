#include "common-sdl.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_set>
#include <mutex>

audio_async::audio_async(int len_ms) {
    m_len_ms = len_ms;

    m_running = false;
}

audio_async::~audio_async() {
    if (m_dev_id_in) {
        SDL_CloseAudioDevice(m_dev_id_in);
    }
}

bool audio_async::init(int capture_id, int sample_rate) {
    SDL_LogSetPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO);

    static std::mutex sdl_audio_init_mutex;
    std::lock_guard<std::mutex> lock(sdl_audio_init_mutex);

    std::vector<std::string> driver_order;
    const char *env_driver = std::getenv("SDL_AUDIODRIVER");
    if (env_driver && env_driver[0] != '\0') {
        driver_order.emplace_back(env_driver);
    }
    // Reasonable fallbacks across common platforms
    const char *fallbacks[] = {"pipewire", "pulse", "alsa", "dsp", "dummy"};
    for (const char *drv : fallbacks) {
        driver_order.emplace_back(drv);
    }
    // de-duplicate while preserving order
    std::unordered_set<std::string> seen;
    std::vector<std::string> drivers;
    for (const auto &d : driver_order) {
        if (seen.insert(d).second) drivers.push_back(d);
    }

    std::string last_error;
    for (const auto &driver : drivers) {
        setenv("SDL_AUDIODRIVER", driver.c_str(), 1);

        // Re-init the audio subsystem for each driver attempt
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        if (SDL_Init(SDL_INIT_AUDIO) < 0) {
            last_error = SDL_GetError();
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't initialize SDL audio with driver '%s': %s\n", driver.c_str(), last_error.c_str());
            continue;
        }

        SDL_SetHintWithPriority(SDL_HINT_AUDIO_RESAMPLING_MODE, "medium", SDL_HINT_OVERRIDE);

        int nDevices = SDL_GetNumAudioDevices(SDL_TRUE);
        if (nDevices < 0) {
            last_error = SDL_GetError();
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "SDL_GetNumAudioDevices failed (driver '%s'): %s\n", driver.c_str(), last_error.c_str());
            // Try next driver
            continue;
        }

        fprintf(stderr, "%s: using SDL_AUDIODRIVER='%s', found %d capture devices:\n", __func__, driver.c_str(), nDevices);
        for (int i = 0; i < nDevices; i++) {
            fprintf(stderr, "%s:    - Capture device #%d: '%s'\n", __func__, i, SDL_GetAudioDeviceName(i, SDL_TRUE));
        }

        SDL_AudioSpec capture_spec_requested;
        SDL_AudioSpec capture_spec_obtained;

        SDL_zero(capture_spec_requested);
        SDL_zero(capture_spec_obtained);

        capture_spec_requested.freq     = sample_rate;
        capture_spec_requested.format   = AUDIO_F32;
        capture_spec_requested.channels = 1;
        capture_spec_requested.samples  = 1024;
        capture_spec_requested.callback = [](void * userdata, uint8_t * stream, int len) {
            audio_async * audio = (audio_async *) userdata;
            audio->callback(stream, len);
        };
        capture_spec_requested.userdata = this;

        if (capture_id >= 0) {
            fprintf(stderr, "%s: attempt to open capture device %d : '%s' ...\n", __func__, capture_id, SDL_GetAudioDeviceName(capture_id, SDL_TRUE));
            m_dev_id_in = SDL_OpenAudioDevice(SDL_GetAudioDeviceName(capture_id, SDL_TRUE), SDL_TRUE, &capture_spec_requested, &capture_spec_obtained, 0);
        } else {
            fprintf(stderr, "%s: attempt to open default capture device ...\n", __func__);
            m_dev_id_in = SDL_OpenAudioDevice(nullptr, SDL_TRUE, &capture_spec_requested, &capture_spec_obtained, 0);
        }

        if (!m_dev_id_in) {
            last_error = SDL_GetError();
            fprintf(stderr, "%s: couldn't open an audio device for capture with driver '%s': %s!\n", __func__, driver.c_str(), last_error.c_str());
            m_dev_id_in = 0;
            // Try next driver
            continue;
        } else {
            fprintf(stderr, "%s: obtained spec for input device (SDL Id = %d):\n", __func__, m_dev_id_in);
            fprintf(stderr, "%s:     - sample rate:       %d\n",                   __func__, capture_spec_obtained.freq);
            fprintf(stderr, "%s:     - format:            %d (required: %d)\n",    __func__, capture_spec_obtained.format,
                    capture_spec_requested.format);
            fprintf(stderr, "%s:     - channels:          %d (required: %d)\n",    __func__, capture_spec_obtained.channels,
                    capture_spec_requested.channels);
            fprintf(stderr, "%s:     - samples per frame: %d\n",                   __func__, capture_spec_obtained.samples);
        }

        m_sample_rate = capture_spec_obtained.freq;
        m_audio.resize((m_sample_rate*m_len_ms)/1000);
        return true;
    }

    fprintf(stderr, "%s: failed to initialize SDL audio capture across all drivers. Last error: %s\n", __func__, last_error.c_str());
    return false;
}

bool audio_async::resume() {
    if (!m_dev_id_in) {
        fprintf(stderr, "%s: no audio device to resume!\n", __func__);
        return false;
    }

    if (m_running) {
        fprintf(stderr, "%s: already running!\n", __func__);
        return false;
    }

    SDL_PauseAudioDevice(m_dev_id_in, 0);

    m_running = true;

    return true;
}

bool audio_async::pause() {
    if (!m_dev_id_in) {
        fprintf(stderr, "%s: no audio device to pause!\n", __func__);
        return false;
    }

    if (!m_running) {
        fprintf(stderr, "%s: already paused!\n", __func__);
        return false;
    }

    SDL_PauseAudioDevice(m_dev_id_in, 1);

    m_running = false;

    return true;
}

bool audio_async::clear() {
    if (!m_dev_id_in) {
        fprintf(stderr, "%s: no audio device to clear!\n", __func__);
        return false;
    }

    if (!m_running) {
        fprintf(stderr, "%s: not running!\n", __func__);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_audio_pos = 0;
        m_audio_len = 0;
    }

    return true;
}

// callback to be called by SDL
void audio_async::callback(uint8_t * stream, int len) {
    if (!m_running) {
        return;
    }

    size_t n_samples = len / sizeof(float);

    if (n_samples > m_audio.size()) {
        n_samples = m_audio.size();

        stream += (len - (n_samples * sizeof(float)));
    }

    //fprintf(stderr, "%s: %zu samples, pos %zu, len %zu\n", __func__, n_samples, m_audio_pos, m_audio_len);

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_audio_pos + n_samples > m_audio.size()) {
            const size_t n0 = m_audio.size() - m_audio_pos;

            memcpy(&m_audio[m_audio_pos], stream, n0 * sizeof(float));
            memcpy(&m_audio[0], stream + n0 * sizeof(float), (n_samples - n0) * sizeof(float));
        } else {
            memcpy(&m_audio[m_audio_pos], stream, n_samples * sizeof(float));
        }
        m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
        m_audio_len = std::min(m_audio_len + n_samples, m_audio.size());
    }
}

void audio_async::get(int ms, std::vector<float> & result) {
    if (!m_dev_id_in) {
        fprintf(stderr, "%s: no audio device to get audio from!\n", __func__);
        return;
    }

    if (!m_running) {
        fprintf(stderr, "%s: not running!\n", __func__);
        return;
    }

    result.clear();

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (ms <= 0) {
            ms = m_len_ms;
        }

        size_t n_samples = (m_sample_rate * ms) / 1000;
        if (n_samples > m_audio_len) {
            n_samples = m_audio_len;
        }

        result.resize(n_samples);

        int s0 = m_audio_pos - n_samples;
        if (s0 < 0) {
            s0 += m_audio.size();
        }

        if (s0 + n_samples > m_audio.size()) {
            const size_t n0 = m_audio.size() - s0;

            memcpy(result.data(), &m_audio[s0], n0 * sizeof(float));
            memcpy(&result[n0], &m_audio[0], (n_samples - n0) * sizeof(float));
        } else {
            memcpy(result.data(), &m_audio[s0], n_samples * sizeof(float));
        }
    }
}

bool sdl_poll_events() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                {
                    return false;
                }
            default:
                break;
        }
    }

    return true;
}

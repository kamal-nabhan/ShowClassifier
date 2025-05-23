#pragma once

#include <vector>
#include <atomic>
#include <string> // For error messages
#include "concurrent_queue.h" // Your thread-safe queue

// Forward declare SDL types if possible, or include SDL.h
// For simplicity and common practice, we include SDL.h here.
#include <SDL2/SDL.h> // Main SDL header

class AudioCapturer {
public:
    // Target sample rate for Whisper.cpp
    static constexpr int TARGET_SAMPLE_RATE = 16000;
    // Number of channels (mono for Whisper.cpp)
    static constexpr int NUM_CHANNELS = 1;
    // Desired audio format (float 32-bit for Whisper.cpp)
    // SDL uses AUDIO_F32SYS for system-endian float32.
    static constexpr SDL_AudioFormat AUDIO_FORMAT = AUDIO_F32SYS;
    // Audio buffer samples (power of 2 is common, e.g., 1024, 2048)
    // This affects latency. Smaller = lower latency, more callbacks.
    static constexpr Uint16 AUDIO_BUFFER_SAMPLES = 1024;


    AudioCapturer(ConcurrentQueue<std::vector<float>>& audio_data_queue);
    ~AudioCapturer();

    AudioCapturer(const AudioCapturer&) = delete;
    AudioCapturer& operator=(const AudioCapturer&) = delete;

    bool start_stream();
    void stop_stream();
    bool is_running() const { return running_.load(); }

private:
    SDL_AudioDeviceID audio_device_id_ = 0; // 0 is an invalid device ID
    ConcurrentQueue<std::vector<float>>& audio_data_queue_;
    std::atomic<bool> running_{false};

    // SDL audio callback function (must be static or a C-style function)
    static void sdl_audio_callback(void *user_data, Uint8 *stream, int len_bytes);
};

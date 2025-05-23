#pragma once

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <whisper.h> // From whisper.cpp installation/submodule
#include "concurrent_queue.h"
#include "transcript_context_builder.h"

class WhisperTranscriber {
public:
    WhisperTranscriber(const std::string& model_path,
                       const std::string& language, // e.g., "en", "auto"
                       ConcurrentQueue<std::vector<float>>& audio_data_queue,
                       TranscriptContextBuilder& context_builder);
    ~WhisperTranscriber();

    WhisperTranscriber(const WhisperTranscriber&) = delete;
    WhisperTranscriber& operator=(const WhisperTranscriber&) = delete;

    bool start();
    void stop();
    bool is_running() const { return running_.load(); }

private:
    // Whisper context and parameters
    struct whisper_context* whisper_ctx_ = nullptr;
    struct whisper_full_params whisper_params_;

    // References to shared resources
    ConcurrentQueue<std::vector<float>>& audio_data_queue_;
    TranscriptContextBuilder& context_builder_;

    // Threading
    std::thread transcribe_thread_;
    std::atomic<bool> running_{false};

    // Internal audio buffer for accumulating samples for whisper.cpp
    std::vector<float> internal_audio_buffer_;

    // Main loop for the transcription thread
    void transcribe_loop();

    // Static callback for whisper.cpp (C-style)
    static void whisper_new_segment_callback(struct whisper_context *ctx, struct whisper_state *state, int n_new, void *user_data);
};
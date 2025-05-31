#include "audio_capturer.h"
#include <iostream>
#include <stdexcept> // For std::runtime_error
#include <vector>

AudioCapturer::AudioCapturer(ConcurrentQueue<std::vector<float>>& audio_data_queue)
    : audio_data_queue_(audio_data_queue) {
    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        throw std::runtime_error(std::string("SDL_Init(SDL_INIT_AUDIO) failed: ") + SDL_GetError());
    }
    std::cout << "[AudioCapturer] SDL Audio initialized." << std::endl;
}

AudioCapturer::~AudioCapturer() {
    stop_stream(); // Ensure stream is stopped and device is closed
    SDL_QuitSubSystem(SDL_INIT_AUDIO); // Quit only the audio subsystem
    // If SDL_Init was called for other subsystems, SDL_Quit() would be used.
    std::cout << "[AudioCapturer] SDL Audio subsystem quit." << std::endl;
}

bool AudioCapturer::start_stream() {
    if (running_) {
        std::cout << "[AudioCapturer] Audio stream is already running." << std::endl;
        return true;
    }

    SDL_AudioSpec desired_spec, obtained_spec;

    SDL_zero(desired_spec); // Important to zero out the structure
    desired_spec.freq = TARGET_SAMPLE_RATE;
    desired_spec.format = AUDIO_FORMAT; // AUDIO_F32SYS
    desired_spec.channels = NUM_CHANNELS;
    desired_spec.samples = AUDIO_BUFFER_SAMPLES; // Size of buffer in samples
    desired_spec.callback = AudioCapturer::sdl_audio_callback;
    desired_spec.userdata = this; // Pass pointer to this AudioCapturer instance

    // Open the default recording device
    // The first parameter being NULL means default recording device.
    // The '1' means it's for recording (iscapture=true).
    audio_device_id_ = SDL_OpenAudioDevice(nullptr, 1, &desired_spec, &obtained_spec, 0);
                                        // The last '0' means no changes allowed to desired_spec.
                                        // Use SDL_AUDIO_ALLOW_FREQUENCY_CHANGE etc. if you want to allow changes.

    if (audio_device_id_ == 0) {
        std::cerr << "[AudioCapturer] SDL_OpenAudioDevice failed: " << SDL_GetError() << std::endl;
        return false;
    }

    // You can check obtained_spec here if you allowed changes to see what format you actually got.
    // For this example, we assume we got what we asked for or SDL handled it.
    // E.g., if (obtained_spec.format != desired_spec.format) { /* handle discrepancy */ }

    SDL_PauseAudioDevice(audio_device_id_, 0); // 0 to unpause (start audio callback)

    running_ = true;
    std::cout << "SDL Audio stream started successfully with device ID: " << audio_device_id_ << std::endl;
    std::cout << "  Frequency: " << obtained_spec.freq << " Hz" << std::endl;
    std::cout << "  Format: " << obtained_spec.format << " (Target was " << AUDIO_FORMAT << ")" << std::endl;
    std::cout << "  Channels: " << (int)obtained_spec.channels << std::endl;
    std::cout << "  Samples per callback: " << obtained_spec.samples << std::endl;
    return true;
}

void AudioCapturer::stop_stream() {
    if (!running_ || audio_device_id_ == 0) {
        return;
    }

    running_ = false; // Signal callback to stop processing (important for thread safety)

    SDL_PauseAudioDevice(audio_device_id_, 1); // 1 to pause the device
    SDL_CloseAudioDevice(audio_device_id_);
    audio_device_id_ = 0; // Mark as closed

    std::cout << "SDL Audio stream stopped." << std::endl;
}

// Static SDL audio callback
void AudioCapturer::sdl_audio_callback(void *user_data, Uint8 *raw_buffer, int len_bytes) {
    AudioCapturer* self = static_cast<AudioCapturer*>(user_data);

    if (!self || !self->running_) {
        // If not running, or self is null, do nothing or fill with silence if it were playback
        // For capture, we just don't push to the queue.
        // SDL_memset(raw_buffer, 0, len_bytes); // Not strictly necessary for capture only
        return;
    }

    // The raw_buffer contains audio data in the format specified by obtained_spec.
    // We requested AUDIO_F32SYS, so it should be float samples.
    // Number of float samples = len_bytes / sizeof(float)
    int num_samples = len_bytes / sizeof(float);
    if (num_samples <= 0) {
        return;
    }

    // Cast the raw buffer to float*
    std::cout << "[AudioCapturer DEBUG] sdl_audio_callback called, len_bytes: " << len_bytes << ", num_samples: " << num_samples << std::endl;
    float* float_buffer = reinterpret_cast<float*>(raw_buffer);

    // Create a vector and copy the data
    std::vector<float> audio_chunk(float_buffer, float_buffer + num_samples);

    // Push to the concurrent queue for the WhisperTranscriber
    self->audio_data_queue_.push(std::move(audio_chunk));
}
// Note: The queue should be thread-safe and handle the data appropriately.
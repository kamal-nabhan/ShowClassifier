#include "whisper_transcriber.h"
#include "audio_capturer.h" // For AudioCapturer::TARGET_SAMPLE_RATE
#include <iostream>
#include <stdexcept> // For std::runtime_error
#include <algorithm> // For std::max

WhisperTranscriber::WhisperTranscriber(
    const std::string& model_path,
    const std::string& language,
    ConcurrentQueue<std::vector<float>>& audio_data_queue,
    TranscriptContextBuilder& context_builder)
    : audio_data_queue_(audio_data_queue),
      context_builder_(context_builder) {

    // Initialize whisper context parameters
    struct whisper_context_params cparams = whisper_context_default_params();
    // cparams.use_gpu = false; // Set to true if you want to try GPU and have it compiled with GPU support

    // Initialize whisper context using the new function
    whisper_ctx_ = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (whisper_ctx_ == nullptr) {
        throw std::runtime_error("Failed to initialize whisper.cpp context from model: " + model_path);
    }
    std::cout << "Whisper.cpp context initialized with model: " << model_path << std::endl;

    // Setup whisper parameters for full transcription
    whisper_params_ = whisper_full_default_params(WHISPER_SAMPLING_GREEDY); // Or WHISPER_SAMPLING_BEAM_SEARCH
    
    whisper_params_.language = language.c_str(); // e.g. "en", "es", "auto"
    whisper_params_.translate = false;           // Transcribe in the original language
    
    // Disable whisper.cpp's own console output, as we'll handle it
    whisper_params_.print_special = false;
    whisper_params_.print_progress = false;
    whisper_params_.print_realtime = false;
    whisper_params_.print_timestamps = false; // Set true if you want timestamps from whisper.cpp

    // Number of threads for whisper.cpp to use for processing
    int num_threads = std::max(1, (int)std::thread::hardware_concurrency() / 2);
    whisper_params_.n_threads = num_threads > 0 ? num_threads : 4; 

    // Set the callback for new segments
    whisper_params_.new_segment_callback = WhisperTranscriber::whisper_new_segment_callback;
    whisper_params_.new_segment_callback_user_data = this; 
}

WhisperTranscriber::~WhisperTranscriber() {
    stop(); 
    if (whisper_ctx_ != nullptr) {
        whisper_free(whisper_ctx_);
        whisper_ctx_ = nullptr;
        std::cout << "Whisper.cpp context freed." << std::endl;
    }
}

bool WhisperTranscriber::start() {
    if (running_) {
        std::cout << "WhisperTranscriber is already running." << std::endl;
        return true;
    }
    running_ = true;
    transcribe_thread_ = std::thread(&WhisperTranscriber::transcribe_loop, this);
    std::cout << "WhisperTranscriber started." << std::endl;
    return true;
}

void WhisperTranscriber::stop() {
    if (!running_) {
        return;
    }
    running_ = false;
    if (transcribe_thread_.joinable()) {
        transcribe_thread_.join();
    }
    std::cout << "WhisperTranscriber stopped." << std::endl;
}

// Static callback function
void WhisperTranscriber::whisper_new_segment_callback(
    struct whisper_context *ctx, struct whisper_state *state, int n_new, void *user_data) {
    
    WhisperTranscriber* self = static_cast<WhisperTranscriber*>(user_data);
    if (!self || !self->running_) {
        return;
    }

    const int n_segments = whisper_full_n_segments(ctx); 
    std::string new_text_accumulated;

    for (int i = n_segments - n_new; i < n_segments; ++i) {
        const char* segment_text = whisper_full_get_segment_text(ctx, i);
        if (segment_text) {
            new_text_accumulated += segment_text;
        }
    }

    if (!new_text_accumulated.empty()) {
        self->context_builder_.append_text(new_text_accumulated + " ");
    }
}


void WhisperTranscriber::transcribe_loop() {
    const size_t SAMPLES_TO_ACCUMULATE_BEFORE_PROCESSING = AudioCapturer::TARGET_SAMPLE_RATE * 5; 
    internal_audio_buffer_.reserve(SAMPLES_TO_ACCUMULATE_BEFORE_PROCESSING + (AudioCapturer::TARGET_SAMPLE_RATE * 2)); 

    while (running_) {
        std::vector<float> new_audio_chunk;
        if (audio_data_queue_.try_pop_for(new_audio_chunk, std::chrono::milliseconds(100))) {
            internal_audio_buffer_.insert(internal_audio_buffer_.end(), new_audio_chunk.begin(), new_audio_chunk.end());
        } else {
            if (!running_) break;
            continue; 
        }

        if (internal_audio_buffer_.size() >= SAMPLES_TO_ACCUMULATE_BEFORE_PROCESSING || 
            (!running_ && !internal_audio_buffer_.empty())) {
            
            std::vector<float> chunk_to_process;
            size_t num_samples_to_process = internal_audio_buffer_.size();
            
            if (num_samples_to_process > 0) {
                 chunk_to_process.assign(internal_audio_buffer_.begin(), internal_audio_buffer_.end());
                 internal_audio_buffer_.clear(); 
            
                int ret = whisper_full(whisper_ctx_, whisper_params_, chunk_to_process.data(), chunk_to_process.size());
                if (ret != 0) {
                    std::cerr << "Whisper_full failed with code: " << ret << std::endl;
                }
            }
        }
        if (!running_ && internal_audio_buffer_.empty()) {
            break;
        }
    }
    if (!internal_audio_buffer_.empty()) {
        int ret = whisper_full(whisper_ctx_, whisper_params_, internal_audio_buffer_.data(), internal_audio_buffer_.size());
        if (ret != 0) {
            std::cerr << "Whisper_full (final flush) failed with code: " << ret << std::endl;
        }
        internal_audio_buffer_.clear();
    }
}

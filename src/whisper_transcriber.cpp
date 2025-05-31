#include "whisper_transcriber.h"
#include "audio_capturer.h" 
#include <iostream>
#include <stdexcept> 
#include <algorithm> 

WhisperTranscriber::WhisperTranscriber(
    const std::string& model_path,
    const std::string& language,
    ConcurrentQueue<std::vector<float>>& audio_data_queue,
    TranscriptContextBuilder& context_builder)
    : audio_data_queue_(audio_data_queue),
      context_builder_(context_builder) {

    std::cout << "[WhisperTranscriber] Initializing..." << std::endl;
    struct whisper_context_params cparams = whisper_context_default_params();
    // cparams.use_gpu = true; // Enable if you have a GPU build and want to use it

    whisper_ctx_ = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (whisper_ctx_ == nullptr) {
        throw std::runtime_error("[WhisperTranscriber] Failed to initialize whisper.cpp context from model: " + model_path);
    }
    std::cout << "[WhisperTranscriber] Whisper.cpp context initialized with model: " << model_path << std::endl;

    whisper_params_ = whisper_full_default_params(WHISPER_SAMPLING_GREEDY); 
    
    whisper_params_.language = language.c_str(); 
    whisper_params_.translate = false;           
    
    whisper_params_.print_special = false;
    whisper_params_.print_progress = false;
    whisper_params_.print_realtime = false;
    whisper_params_.print_timestamps = false; 

    int num_threads = std::max(1, (int)std::thread::hardware_concurrency() / 2);
    num_threads = num_threads > 0 ? num_threads : 4; // Ensure at least 1, default to 4 if detection fails weirdly
    whisper_params_.n_threads = num_threads;
    std::cout << "[WhisperTranscriber] Using " << num_threads << " threads for transcription." << std::endl;

    whisper_params_.new_segment_callback = WhisperTranscriber::whisper_new_segment_callback;
    whisper_params_.new_segment_callback_user_data = this; 
    std::cout << "[WhisperTranscriber] Initialization complete." << std::endl;
}

WhisperTranscriber::~WhisperTranscriber() {
    std::cout << "[WhisperTranscriber] Destructing..." << std::endl;
    stop(); 
    if (whisper_ctx_ != nullptr) {
        whisper_free(whisper_ctx_);
        whisper_ctx_ = nullptr;
        std::cout << "[WhisperTranscriber] Whisper.cpp context freed." << std::endl;
    }
}

bool WhisperTranscriber::start() {
    if (running_) {
        std::cout << "[WhisperTranscriber] is already running." << std::endl;
        return true;
    }
    running_ = true;
    transcribe_thread_ = std::thread(&WhisperTranscriber::transcribe_loop, this);
    std::cout << "[WhisperTranscriber] Started transcription thread." << std::endl;
    return true;
}

void WhisperTranscriber::stop() {
    if (!running_) {
        return;
    }
    std::cout << "[WhisperTranscriber] Stopping transcription thread..." << std::endl;
    running_ = false;
    if (transcribe_thread_.joinable()) {
        transcribe_thread_.join();
    }
    std::cout << "[WhisperTranscriber] Transcription thread stopped." << std::endl;
}

void WhisperTranscriber::whisper_new_segment_callback(
    struct whisper_context *ctx, struct whisper_state *state, int n_new, void *user_data) {
    
    WhisperTranscriber* self = static_cast<WhisperTranscriber*>(user_data);
    if (!self || !self->running_) {
        return;
    }

    const int n_segments = whisper_full_n_segments(ctx); 
    std::string new_text_accumulated;

    //std::cout << "[WhisperTranscriber DEBUG] New segment callback: n_new = " << n_new << ", total_segments = " << n_segments << std::endl;

    for (int i = std::max(0, n_segments - n_new); i < n_segments; ++i) { // Ensure 'i' starts from at least 0
        const char* segment_text = whisper_full_get_segment_text(ctx, i);
        if (segment_text) {
            new_text_accumulated += segment_text;
           // std::cout << "[WhisperTranscriber DEBUG] Segment " << i << ": " << segment_text << std::endl;
        }
    }

    if (!new_text_accumulated.empty()) {
        std::cout << "[WhisperTranscriber] New transcript segment: \"" << new_text_accumulated << "\"" << std::endl;
        self->context_builder_.append_text(new_text_accumulated + " "); // Add space between segments
    }
}


void WhisperTranscriber::transcribe_loop() {
    const size_t SAMPLES_TO_ACCUMULATE_BEFORE_PROCESSING = AudioCapturer::TARGET_SAMPLE_RATE * 5; // 5 seconds of audio
    // Reduced accumulation for faster feedback during debugging, can be increased later
    // const size_t SAMPLES_TO_ACCUMULATE_BEFORE_PROCESSING = AudioCapturer::TARGET_SAMPLE_RATE * 2; // 2 seconds

    internal_audio_buffer_.reserve(SAMPLES_TO_ACCUMULATE_BEFORE_PROCESSING + (AudioCapturer::TARGET_SAMPLE_RATE * 2)); 
    std::cout << "[WhisperTranscriber] Transcribe loop started. Waiting for at least " << SAMPLES_TO_ACCUMULATE_BEFORE_PROCESSING << " samples." << std::endl;

    while (running_) {
        std::vector<float> new_audio_chunk;
        if (audio_data_queue_.try_pop_for(new_audio_chunk, std::chrono::milliseconds(100))) {
            //std::cout << "[WhisperTranscriber DEBUG] Popped " << new_audio_chunk.size() << " samples from queue." << std::endl;
            internal_audio_buffer_.insert(internal_audio_buffer_.end(), new_audio_chunk.begin(), new_audio_chunk.end());
            //std::cout << "[WhisperTranscriber DEBUG] Internal buffer size: " << internal_audio_buffer_.size() << " samples." << std::endl;
        } else {
            if (!running_) break;
            // No data, sleep briefly or continue to allow running_ flag to be checked
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Sleep a bit if queue is empty but still running
            continue; 
        }

        if (internal_audio_buffer_.size() >= SAMPLES_TO_ACCUMULATE_BEFORE_PROCESSING) {
            std::cout << "[WhisperTranscriber] Accumulated " << internal_audio_buffer_.size() << " samples. Processing..." << std::endl;
            
            // Copy data to process and clear the internal buffer for new audio.
            // This specific approach to copy and clear might need review for optimal performance
            // but should be functionally correct for now.
            std::vector<float> chunk_to_process = internal_audio_buffer_; // Make a copy
            internal_audio_buffer_.clear(); // Clear for next accumulation
            
            //std::cout << "[WhisperTranscriber DEBUG] Processing chunk of size: " << chunk_to_process.size() << std::endl;
            int ret = whisper_full(whisper_ctx_, whisper_params_, chunk_to_process.data(), chunk_to_process.size());
            if (ret != 0) {
                std::cerr << "[WhisperTranscriber] ERROR: whisper_full failed with code: " << ret << std::endl;
            } else {
                //std::cout << "[WhisperTranscriber DEBUG] whisper_full call successful." << std::endl;
                // New segments are handled by the callback
            }
        }
         // Check running_ flag again to ensure timely exit if stop() was called during processing
        if (!running_ && internal_audio_buffer_.empty()) {
            break;
        }
    }
    
    // Process any remaining audio when stopping
    if (!internal_audio_buffer_.empty()) {
        std::cout << "[WhisperTranscriber] Processing remaining " << internal_audio_buffer_.size() << " samples before exiting." << std::endl;
        int ret = whisper_full(whisper_ctx_, whisper_params_, internal_audio_buffer_.data(), internal_audio_buffer_.size());
        if (ret != 0) {
            std::cerr << "[WhisperTranscriber] ERROR: whisper_full (final flush) failed with code: " << ret << std::endl;
        }
        internal_audio_buffer_.clear();
    }
    std::cout << "[WhisperTranscriber] Transcribe loop finished." << std::endl;
}
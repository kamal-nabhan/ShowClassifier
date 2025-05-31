#include "transcript_context_builder.h"
#include <iostream> // For logging

void TranscriptContextBuilder::append_text(const std::string& text) {
    std::lock_guard<std::mutex> lock(mutex_);
    cumulative_transcript_ += text;
    //std::cout << "[TranscriptContextBuilder DEBUG] Appended text. Current full transcript length: " << cumulative_transcript_.length() << std::endl;
    //std::cout << "[TranscriptContextBuilder DEBUG] Current Transcript: \"" << cumulative_transcript_ << "\"" << std::endl;
}

std::string TranscriptContextBuilder::get_full_transcript() {
    std::lock_guard<std::mutex> lock(mutex_);
    //std::cout << "[TranscriptContextBuilder DEBUG] get_full_transcript() called. Length: " << cumulative_transcript_.length() << std::endl;
    return cumulative_transcript_; 
}

void TranscriptContextBuilder::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cumulative_transcript_.clear();
    std::cout << "[TranscriptContextBuilder] Cleared transcript." << std::endl;
}
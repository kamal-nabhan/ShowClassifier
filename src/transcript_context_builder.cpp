#include "transcript_context_builder.h"

void TranscriptContextBuilder::append_text(const std::string& text) {
    std::lock_guard<std::mutex> lock(mutex_);
    cumulative_transcript_ += text;
}

std::string TranscriptContextBuilder::get_full_transcript() {
    std::lock_guard<std::mutex> lock(mutex_);
    return cumulative_transcript_; // Return a copy
}

void TranscriptContextBuilder::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cumulative_transcript_.clear();
}
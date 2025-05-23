#pragma once

#include <string>
#include <mutex>

class TranscriptContextBuilder {
public:
    TranscriptContextBuilder() = default;
    TranscriptContextBuilder(const TranscriptContextBuilder&) = delete;
    TranscriptContextBuilder& operator=(const TranscriptContextBuilder&) = delete;

    void append_text(const std::string& text);
    std::string get_full_transcript(); // Gets a copy of the current transcript
    void clear();

private:
    std::string cumulative_transcript_;
    mutable std::mutex mutex_; // Protects cumulative_transcript_
};
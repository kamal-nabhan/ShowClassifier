#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include "transcript_context_builder.h"

class OpenAIClient {
public:
    OpenAIClient(TranscriptContextBuilder& context_builder, std::string api_key, const std::string& model_name = "gpt-3.5-turbo");
    ~OpenAIClient();

    OpenAIClient(const OpenAIClient&) = delete;
    OpenAIClient& operator=(const OpenAIClient&) = delete;

    void start_periodic_classification(std::chrono::seconds interval);
    void stop_classification_loop();
    bool is_running() const { return running_.load(); }
    std::string get_last_classification_result() const;

private:
    TranscriptContextBuilder& context_builder_;
    std::string api_key_;
    std::string model_name_;
    std::string system_prompt_ = "You are an expert at identifying movies and TV shows from dialogue transcripts. The transcript provided may be in one of several languages. Your task is to identify the movie or TV show. If unsure, respond with 'Unknown'.";
    
    std::string last_classification_result_ = "Unknown";
    mutable std::mutex result_mutex_; // To protect last_classification_result_

    // Threading for periodic calls
    std::thread classification_thread_;
    std::atomic<bool> running_{false};
    std::chrono::seconds classification_interval_{10}; // Default interval

    void periodic_classification_loop();
    std::string classify_text_with_openai(const std::string& transcript_text);
    std::string parse_openai_json_response(const std::string& json_response_str);
};
#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include "transcript_context_builder.h"

class OpenAIClient {
public:
    OpenAIClient(TranscriptContextBuilder& context_builder, 
                 std::string api_key, 
                 const std::string& openai_endpoint,
                 const std::string& deployment_name, // Renamed from model_name for clarity with Azure
                 const std::string& api_version);
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
    std::string openai_endpoint_;
    std::string deployment_name_; // Used as model_name in payload for some APIs, but primarily for URL with Azure
    std::string api_version_;

    // MODIFICATION 1: Updated system prompt
    std::string system_prompt_ = R"(You are a media recognition expert with deep knowledge of movies and TV shows. I will provide a transcript of dialogue. Your task is to identify whether the dialogue is from a **movie** or a **TV show**.
### If the dialogue is from a **movie**, return:
- Full movie title  
- Release year  
- Specific part or version (if applicable, e.g., 'Part 2', 'Director's Cut', 'Remake (2019)', etc.)
### If the dialogue is from a **TV show**, return:
- TV show title  
- Season number  
- Episode number  
- Episode title (if available)
Be specific and accurate, especially for media with multiple adaptations, remakes, or sequels.
---
Here is the dialogue transcript: )"; // The STT output will be appended by the calling code.
    
    std::string last_classification_result_ = "Unknown";
    mutable std::mutex result_mutex_; // To protect last_classification_result_

    // Threading for periodic calls
    std::thread classification_thread_;
    std::atomic<bool> running_{false};
    std::chrono::seconds classification_interval_{10}; // Default interval

    void periodic_classification_loop();
    std::string classify_text_with_openai(const std::string& transcript_text); // Transcript text will be appended to system_prompt_
    std::string parse_openai_json_response(const std::string& json_response_str);
};
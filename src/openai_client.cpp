#include "openai_client.h"
#include <iostream>
#include <stdexcept> // For std::runtime_error
#include <string_view> // For string manipulation if needed

// CPR for HTTP requests - ensure you have this library available and linked
// #define CPR_USE_SYSTEM_CURL // Optional: if you want CPR to use system's libcurl
#include <cpr/cpr.h>

// nlohmann/json for JSON manipulation - ensure you have this library available
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// MODIFICATION 2: Updated constructor to accept Azure specific parameters
OpenAIClient::OpenAIClient(TranscriptContextBuilder& context_builder, 
                           std::string api_key, 
                           const std::string& openai_endpoint,
                           const std::string& deployment_name,
                           const std::string& api_version)
    : context_builder_(context_builder), 
      api_key_(std::move(api_key)),
      openai_endpoint_(openai_endpoint),
      deployment_name_(deployment_name),
      api_version_(api_version) {
    if (api_key_.empty()) {
        std::cerr << "Warning: OpenAI API key is empty. Classification will fail." << std::endl;
    }
    if (openai_endpoint_.empty()) {
        std::cerr << "Warning: OpenAI endpoint is empty. Classification will fail." << std::endl;
    }
    if (deployment_name_.empty()) {
        std::cerr << "Warning: OpenAI deployment name is empty. Classification will fail." << std::endl;
    }
    if (api_version_.empty()) {
        std::cerr << "Warning: OpenAI API version is empty. Classification will fail." << std::endl;
    }
}

OpenAIClient::~OpenAIClient() {
    stop_classification_loop();
}

void OpenAIClient::start_periodic_classification(std::chrono::seconds interval) {
    if (running_) {
        std::cout << "OpenAIClient classification loop is already running." << std::endl;
        return;
    }
    if (api_key_.empty() || openai_endpoint_.empty() || deployment_name_.empty() || api_version_.empty()) {
        std::cerr << "Cannot start OpenAIClient: API key, endpoint, deployment name, or API version is empty." << std::endl;
        return;
    }
    classification_interval_ = interval;
    running_ = true;
    classification_thread_ = std::thread(&OpenAIClient::periodic_classification_loop, this);
    std::cout << "OpenAIClient periodic classification started with interval: " << interval.count() << "s." << std::endl;
}

void OpenAIClient::stop_classification_loop() {
    if (!running_) {
        return;
    }
    running_ = false;
    if (classification_thread_.joinable()) {
        classification_thread_.join();
    }
    std::cout << "OpenAIClient classification loop stopped." << std::endl;
}

std::string OpenAIClient::get_last_classification_result() const {
    std::lock_guard<std::mutex> lock(result_mutex_);
    return last_classification_result_;
}

void OpenAIClient::periodic_classification_loop() {
    while (running_) {
        std::this_thread::sleep_for(classification_interval_);
        if (!running_) break;

        std::string current_transcript = context_builder_.get_full_transcript();
        if (current_transcript.empty() || current_transcript.find_first_not_of(" \n\r\t") == std::string::npos) {
            continue;
        }
        std::cout << "==> OpenAIClient: Classifying transcript..." << current_transcript<< std::endl;
        
        // The system_prompt_ already has the placeholder text format.
        // We just need to append the actual transcript.
        // The prompt now looks like: "System prompt instructions... Here is the dialogue transcript: <actual transcript>"
        std::string full_prompt_for_user_role = system_prompt_ + current_transcript;


        // For Azure, the system prompt is typically sent as a separate message with "role":"system"
        // and the user message (transcript) as another message with "role":"user".
        // The current system_prompt_ is designed to be the content of the system role message.
        // The user message will be just the transcript.

        std::string classification_result = classify_text_with_openai(current_transcript); // Pass only the transcript
        
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            last_classification_result_ = classification_result;
        }
        std::cout << "==> OpenAI Classification: " << classification_result << std::endl;
    }
}

std::string OpenAIClient::classify_text_with_openai(const std::string& transcript_text) {
    // The system_prompt_ member variable holds the main instructions.
    // The transcript_text is the dialogue.
    
    // Prepare the JSON payload for the OpenAI API request
    // The system_prompt_ already includes "Here is the dialogue transcript: "
    // So the user message will just be the transcript_text.
    json payload = {
        // For Azure, the 'model' is the deployment_name specified in the URL.
        // Some Azure versions might still expect/allow 'model' in payload, but it's often ignored if deployment is in URL.
        // We will omit it here as it's part of the URL.
        {"messages", json::array({
            // MODIFICATION 1: System prompt is used as the content for the "system" role message.
            {{"role", "system"}, {"content", system_prompt_}}, 
            // The transcript_text is the user's message.
            {{"role", "user"}, {"content", transcript_text}} 
        })},
        {"temperature", 0.2},
        {"max_tokens", 150} // Increased max_tokens slightly to accommodate potentially longer structured responses
    };

    // MODIFICATION 2: Construct Azure OpenAI URL
    // Format: {endpoint}/openai/deployments/{deployment-name}/chat/completions?api-version={api-version}
    std::string request_url = openai_endpoint_;
    if (request_url.back() != '/') {
        request_url += '/';
    }
    request_url += "openai/deployments/" + deployment_name_ + "/chat/completions?api-version=" + api_version_;

    try {
        cpr::Response r = cpr::Post(cpr::Url{request_url},
                                    // MODIFICATION 2: Use api-key header for Azure
                                    cpr::Header{{"Content-Type", "application/json"},
                                                {"api-key", api_key_}},
                                    cpr::Body{payload.dump()},
                                    cpr::Timeout{15000}); // 15 seconds timeout

        if (r.status_code == 200) {
            return parse_openai_json_response(r.text);
        } else {
            std::cerr << "[OpenAIClient] API Error " << r.status_code << ": " << r.text << std::endl;
            std::cerr << "[OpenAIClient] Request URL: " << request_url << std::endl;
            std::cerr << "[OpenAIClient] Payload: " << payload.dump(2) << std::endl;
            if (r.status_code == 401) return "Unknown (Invalid API Key or Auth Error)";
            if (r.status_code == 404) return "Unknown (Endpoint or Deployment Not Found)";
            if (r.status_code == 429) return "Unknown (Rate Limit Exceeded)";
            return std::string("Unknown (API Error ") + std::to_string(r.status_code) + ")";
        }
    } catch (const std::exception& e) {
        std::cerr << "[OpenAIClient] HTTP Request Exception: " << e.what() << std::endl;
        return "Unknown (Request Exception)";
    }
}

std::string OpenAIClient::parse_openai_json_response(const std::string& json_response_str) {
    try {
        json response_json = json::parse(json_response_str);
        if (response_json.contains("choices") && 
            response_json["choices"].is_array() && 
            !response_json["choices"].empty()) {
            
            const auto& first_choice = response_json["choices"][0];
            if (first_choice.contains("message") && 
                first_choice["message"].contains("content")) {
                return first_choice["message"]["content"].get<std::string>();
            }
        }
        std::cerr << "[OpenAIClient] Could not find 'content' in choices[0].message. Response: " << json_response_str << std::endl;
        return "Unknown (Invalid Response Format)";
    } catch (const json::parse_error& e) {
        std::cerr << "[OpenAIClient] JSON parse error: " << e.what() << ". Response was: " << json_response_str << std::endl;
        return "Unknown (Response Parse Error)";
    } catch (const std::exception& e) {
        std::cerr << "[OpenAIClient] Error parsing JSON: " << e.what() << std::endl;
        return "Unknown (Generic Parse Error)";
    }
}
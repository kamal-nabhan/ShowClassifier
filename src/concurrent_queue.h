#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono> // For try_pop_for timeout

template <typename T>
class ConcurrentQueue {
public:
    ConcurrentQueue() = default;
    ConcurrentQueue(const ConcurrentQueue&) = delete; // Disable copying
    ConcurrentQueue& operator=(const ConcurrentQueue&) = delete; // Disable assignment

    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        cv_.notify_one();
    }

    // Waits until an item is available
    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    // Tries to pop an item without waiting, returns true if successful
    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    // Tries to pop an item, waiting up to a specified duration
    bool try_pop_for(T& item, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            // Timeout occurred and queue is still empty
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};
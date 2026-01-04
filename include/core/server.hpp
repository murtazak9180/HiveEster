#pragma once
#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include <queue>


using Eigen::VectorXd;
using Eigen::MatrixXd;
const int PORT = 8080;
const int BUFFER_SIZE = 1024; 
const int FEATURE_DIM = 10; 

enum ClientState{
    CONNECTED, 
    READY, 
    TRAINING, 
    DISCONNECTED
};

struct ClientContext{
    uint64_t client_id; 
    int sockfd; 
    std::thread::id thread_id; 
    ClientState state; 
    std::chrono::steady_clock::time_point last_seen;
};

struct ServerState{
    VectorXd global_weights(FEATURE_DIM);
    std::unordered_map<uint64_t, ClientContext> client_registry; 
    std::queue<uint64_t> client_queue; 
    std::mutex registery_mutex; 
    std::mutex queue_mutex; 
    std::condition_variable clients_ready_cv; 
};



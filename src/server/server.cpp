#include <condition_variable>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <queue>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include <utility>



#include "../../include/core/server.hpp"


using Eigen::VectorXd;
using Eigen::MatrixXd;



void accept_loop(ServerState* st){

}

void round_manager_loop(ServerState* st){

}


int main(){
    int server_fd, new_socket; 
    struct sockaddr_in addr; 
    int opt = 1; 
    int addrlen = sizeof(addr); 

    if((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0){
        perror("Socket initialization failed");
        exit(EXIT_FAILURE); 
    }

    addr.sin_family = AF_INET; 
    addr.sin_addr.s_addr = INADDR_ANY; 
    addr.sin_port = htons(PORT); 

    if(bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0){
        perror("Bind failed");
        exit(EXIT_FAILURE); 
    }


    if(listen(server_fd, 5)<0){
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }

    std::cout<< "Server listening on port: " << PORT << std::endl; 

    ServerState st; 
    
    std::thread round_manager(round_manager_loop, &st);
    std::thread accept_thread(accept_loop, &st);

    // wait for the round manager to finish before exiting so `st` remains valid
    // if(round_manager.joinable()){
    //     round_manager.join();
    // }
    // if(accept_thread.joinable()){
    //     accept_thread.join();
    // }

    close(server_fd);
    return 0; 
}
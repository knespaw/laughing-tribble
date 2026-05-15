#include <iostream>
#include "llama.h" // The core C API for llama.cpp


int main() {
    std::cout << "Starting LLM Agent Project..." << std::endl;

    // 1. Initialize the backend.
    // The "true" argument tells it to enable NUMA (not strictly needed for Mac, but good practice).
    llama_backend_init();

    // 2. Query the system info to prove Metal is enabled
    std::cout << "\n=== System Info ===" << std::endl;
    std::cout << llama_print_system_info() << std::endl;

    // 3. Clean up
    llama_backend_free();

    std::cout << "\nSetup successful! Ready to load models." << std::endl;
    return 0;
}

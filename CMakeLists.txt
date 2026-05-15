cmake_minimum_required(VERSION 4.0)
project(laughing_tribble)

set(CMAKE_CXX_STANDARD 23)

set(GGML_METAL ON CACHE BOOL "Enable Apple Metal GPU acceleration" FORCE)

add_subdirectory(dependencies/llama.cpp)

add_executable(laughing_tribble main.cpp)

target_link_libraries(laughing_tribble PRIVATE llama)

target_include_directories(laughing_tribble PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/llama.cpp/include)

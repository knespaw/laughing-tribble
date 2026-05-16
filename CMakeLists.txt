### General
cmake_minimum_required(VERSION 4.0)
project(xprv)

set(CMAKE_CXX_STANDARD 23)

set(GGML_METAL ON CACHE BOOL "Enable Apple Metal GPU acceleration" FORCE)


### Deps

# llama.cpp
add_subdirectory(dependencies/llama.cpp)

# spdlog
include(FetchContent)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.17.0
)

FetchContent_MakeAvailable(spdlog)


### Execs
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS "src/*.h")

add_executable(xprv ${SOURCE_FILES} ${HEADER_FILES})


### Linking
target_link_libraries(xprv PRIVATE
    llama
    spdlog::spdlog
)

target_include_directories(xprv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/llms
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
)


### Defs

#
add_definitions(-DQWEN_FILE="${CMAKE_CURRENT_SOURCE_DIR}/model_files/Qwen3-8B-Q5_K_M.gguf")

#
add_definitions(-DLOG_LEVEL=1 -DLOG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/logs") # todo

cmake_minimum_required(VERSION 3.16)
project(riderless_worker LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LLAMA_SOURCE "" CACHE PATH "Frozen llama.cpp source snapshot")
set(LLAMA_BUILD "" CACHE PATH "Matching frozen llama.cpp build")
if(NOT EXISTS "${LLAMA_SOURCE}/include/llama.h")
    message(FATAL_ERROR "Set LLAMA_SOURCE and LLAMA_BUILD to the frozen runtime")
endif()

set(LLAMA_LIB "${LLAMA_BUILD}/bin")
foreach(name llama llama-common ggml ggml-base ggml-cpu)
    find_library(LIB_${name} NAMES ${name} PATHS "${LLAMA_LIB}" NO_DEFAULT_PATH REQUIRED)
endforeach()

add_executable(riderless-worker
    worker.cpp
    "${LLAMA_SOURCE}/examples/gguf-hash/deps/sha256/sha256.c")
target_include_directories(riderless-worker SYSTEM PRIVATE
    "${LLAMA_SOURCE}/include"
    "${LLAMA_SOURCE}/src"
    "${LLAMA_SOURCE}/ggml/include"
    "${LLAMA_SOURCE}/common"
    "${LLAMA_SOURCE}/vendor"
    "${LLAMA_SOURCE}/examples/gguf-hash/deps")
target_link_libraries(riderless-worker PRIVATE
    ${LIB_llama-common} ${LIB_llama} ${LIB_ggml} ${LIB_ggml-base})
target_compile_definitions(riderless-worker PRIVATE
    RIDERLESS_BACKEND_DIR="${LLAMA_LIB}")
target_compile_options(riderless-worker PRIVATE -O3 -Wall -Wextra -Werror)
set_target_properties(riderless-worker PROPERTIES BUILD_RPATH "${LLAMA_LIB}")

add_executable(riderless-worker-utils-test worker-utils-test.cpp)
target_compile_options(riderless-worker-utils-test PRIVATE -Wall -Wextra -Werror)

enable_testing()
add_test(NAME worker-utils COMMAND riderless-worker-utils-test)

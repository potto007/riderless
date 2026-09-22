cmake_minimum_required(VERSION 3.16)
project(riderless_worker LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LLAMA_SOURCE "" CACHE PATH "llama.cpp source snapshot of the base runtime")
set(LLAMA_BUILD "" CACHE PATH "Matching llama.cpp build")
if(NOT EXISTS "${LLAMA_SOURCE}/include/llama.h")
    message(FATAL_ERROR "Set LLAMA_SOURCE and LLAMA_BUILD to the base runtime")
endif()

# The bundled sha256 helper moved out of the gguf-hash example and into vendor
# in llama.cpp v0.4.1. The file is unchanged, so either layout links the same.
set(SHA256_ROOT "")
foreach(candidate "vendor/hash" "examples/gguf-hash/deps")
    if(EXISTS "${LLAMA_SOURCE}/${candidate}/sha256/sha256.c")
        set(SHA256_ROOT "${LLAMA_SOURCE}/${candidate}")
        break()
    endif()
endforeach()
if(SHA256_ROOT STREQUAL "")
    message(FATAL_ERROR "No bundled sha256 helper under ${LLAMA_SOURCE}")
endif()

set(LLAMA_LIB "${LLAMA_BUILD}/bin")
foreach(name llama llama-common ggml ggml-base ggml-cpu)
    find_library(LIB_${name} NAMES ${name} PATHS "${LLAMA_LIB}" NO_DEFAULT_PATH REQUIRED)
endforeach()

add_executable(riderless-worker
    worker.cpp
    "${SHA256_ROOT}/sha256/sha256.c")
target_include_directories(riderless-worker SYSTEM PRIVATE
    "${LLAMA_SOURCE}/include"
    "${LLAMA_SOURCE}/src"
    "${LLAMA_SOURCE}/ggml/include"
    "${LLAMA_SOURCE}/common"
    "${LLAMA_SOURCE}/vendor"
    "${SHA256_ROOT}")
target_link_libraries(riderless-worker PRIVATE
    ${LIB_llama-common} ${LIB_llama} ${LIB_ggml} ${LIB_ggml-base})
target_compile_definitions(riderless-worker PRIVATE
    RIDERLESS_BACKEND_DIR="${LLAMA_LIB}")
target_compile_options(riderless-worker PRIVATE -O3 -Wall -Wextra -Werror)
set_target_properties(riderless-worker PROPERTIES BUILD_RPATH "${LLAMA_LIB}")

# Competitor benchmarks need arbitrary token sequences scored at arbitrary
# readout positions on this exact runtime. That is not the worker's protocol,
# so it is a second, separate binary that shares only the linked libraries and
# the context configuration.
add_executable(riderless-so1-probe so1_probe.cpp)
target_include_directories(riderless-so1-probe SYSTEM PRIVATE
    "${LLAMA_SOURCE}/include"
    "${LLAMA_SOURCE}/ggml/include"
    "${LLAMA_SOURCE}/vendor")
target_link_libraries(riderless-so1-probe PRIVATE ${LIB_llama} ${LIB_ggml} ${LIB_ggml-base})
target_compile_definitions(riderless-so1-probe PRIVATE
    RIDERLESS_BACKEND_DIR="${LLAMA_LIB}")
target_compile_options(riderless-so1-probe PRIVATE -O3 -Wall -Wextra -Werror)
set_target_properties(riderless-so1-probe PROPERTIES BUILD_RPATH "${LLAMA_LIB}")

add_executable(riderless-worker-utils-test worker-utils-test.cpp)
target_compile_options(riderless-worker-utils-test PRIVATE -Wall -Wextra -Werror)

enable_testing()
add_test(NAME worker-utils COMMAND riderless-worker-utils-test)

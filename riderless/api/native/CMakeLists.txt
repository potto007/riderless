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

# A bundle is <dir>/build/<binary> beside <dir>/runtime/*.so, so an
# $ORIGIN-relative run path finds the libraries wherever the bundle was
# unpacked and puts no build-machine path into the binary. It covers this
# binary's own DT_NEEDED entries and no more: the llama.cpp libraries carry
# their own DT_RUNPATH, which suppresses run-path inheritance for anything
# they pull in, and rewriting theirs would change the sha256 the manifest
# pins. LD_LIBRARY_PATH, which the API backend exports and which outranks
# DT_RUNPATH, is what covers those transitive loads and the dlopened backends.
#
# BUILD_WITH_INSTALL_RPATH is what keeps the run path to exactly that one
# entry: without it CMake also appends the absolute directory the llama.cpp
# libraries were linked from, which would put the build machine's path into
# the shipped binary and make its sha256 depend on where it was built.
set(RIDERLESS_BUNDLE_RPATH "$ORIGIN/../runtime")
function(riderless_relocatable target)
    set_target_properties(${target} PROPERTIES
        INSTALL_RPATH "${RIDERLESS_BUNDLE_RPATH}"
        BUILD_WITH_INSTALL_RPATH ON)
    target_link_options(${target} PRIVATE "LINKER:--disable-new-dtags")
endfunction()

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
target_compile_options(riderless-worker PRIVATE -O3 -Wall -Wextra -Werror)
riderless_relocatable(riderless-worker)

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
target_compile_options(riderless-so1-probe PRIVATE -O3 -Wall -Wextra -Werror)
riderless_relocatable(riderless-so1-probe)

add_executable(riderless-worker-utils-test worker-utils-test.cpp)
target_compile_options(riderless-worker-utils-test PRIVATE -Wall -Wextra -Werror)

enable_testing()
add_test(NAME worker-utils COMMAND riderless-worker-utils-test)

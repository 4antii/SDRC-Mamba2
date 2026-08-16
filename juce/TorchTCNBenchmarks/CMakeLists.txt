cmake_minimum_required(VERSION 3.18)

project(TorchTCNBenchmarks VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(JUCE_DIR "" CACHE PATH "Path to the JUCE source tree")
if (JUCE_DIR)
    add_subdirectory("${JUCE_DIR}" JUCE)
else()
    find_package(JUCE CONFIG REQUIRED)
endif()

find_package(Torch CONFIG REQUIRED)

juce_add_binary_data(TorchTCNData
    SOURCES
        Resources/uTCN-100_4-10-5_release.pt
        Resources/uTCN-300_4-10-13_release.pt
)

function(add_torch_tcn_plugin TARGET PRODUCT_NAME PLUGIN_CODE MODEL_FILE CONTEXT_SAMPLES)
    juce_add_plugin(${TARGET}
        COMPANY_NAME "SpectralBench"
        PLUGIN_MANUFACTURER_CODE SbcB
        PLUGIN_CODE ${PLUGIN_CODE}
        FORMATS AU VST3
        PRODUCT_NAME "${PRODUCT_NAME}")

    target_sources(${TARGET} PRIVATE
        Source/TorchTCNProcessor.cpp
        Source/TorchTCNProcessor.h
    )

    target_compile_definitions(${TARGET} PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_USE_CAMERA=0
        JUCE_USE_MP3AUDIOFORMAT=0
        JUCE_USE_OGGVORBIS=0
        JUCE_USE_FLAC=0
        JUCE_USE_LAME_AUDIO_FORMAT=0
        JUCE_VST2_PLUGIN_ENABLE=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        TORCH_TCN_MODEL_ORIGINAL="${MODEL_FILE}"
        TORCH_TCN_CONTEXT_SAMPLES=${CONTEXT_SAMPLES}
    )

    target_link_libraries(${TARGET} PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_processors
        juce::juce_audio_basics
        juce::juce_core
        TorchTCNData
        ${TORCH_LIBRARIES}
    )

    target_include_directories(${TARGET} PRIVATE ${TORCH_INCLUDE_DIRS})
    target_compile_options(${TARGET} PRIVATE ${TORCH_CXX_FLAGS})
    if (TORCH_INSTALL_PREFIX)
        set_target_properties(${TARGET} PROPERTIES
            BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
            INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
        )
    endif()
endfunction()

add_torch_tcn_plugin(UTCN100TorchCPUBenchmark "uTCN 100 Torch CPU Benchmark" Tt15
    "uTCN-100_4-10-5_release.pt" 4444)

add_torch_tcn_plugin(UTCN300TorchCPUBenchmark "uTCN 300 Torch CPU Benchmark" Tt33
    "uTCN-300_4-10-13_release.pt" 13332)

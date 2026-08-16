cmake_minimum_required(VERSION 3.15)

project(CPUModelBenchmarks VERSION 0.1.0)

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

add_executable(s4_validate
    Source/s4_validate.cpp
    Source/S4RawModel.h
    Source/TensorArchive.h
)

target_link_libraries(s4_validate PRIVATE
    juce::juce_core
)

target_compile_definitions(s4_validate PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
)

add_executable(s6_validate
    Source/s6_validate.cpp
    Source/S6RawModel.h
    Source/RawModelUtils.h
    Source/TensorArchive.h
)

target_link_libraries(s6_validate PRIVATE
    juce::juce_core
)

target_compile_definitions(s6_validate PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
)

add_executable(lstm_validate
    Source/lstm_validate.cpp
    Source/LstmRawModel.h
    Source/RawModelUtils.h
    Source/TensorArchive.h
)

target_link_libraries(lstm_validate PRIVATE
    juce::juce_core
)

target_compile_definitions(lstm_validate PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
)

add_executable(mamba2_frame_validate
    Source/mamba2_frame_validate.cpp
    Source/Mamba2SpectralRawModel.h
    Source/Mamba2SizedSpectralRawModel.h
    Source/RawModelUtils.h
    Source/TensorArchive.h
)

target_link_libraries(mamba2_frame_validate PRIVATE
    juce::juce_core
)

target_compile_definitions(mamba2_frame_validate PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
)

juce_add_binary_data(CPUModelBenchmarksData
    SOURCES
        Resources/s4_c32_f4_release.json
        Resources/s4_c32_f4_release.bin
        Resources/lstm_raw_32_release.json
        Resources/lstm_raw_32_release.bin
        Resources/s6_16d_release.json
        Resources/s6_16d_release.bin
        Resources/uTCN-100_4-10-5_release.json
        Resources/uTCN-100_4-10-5_release.bin
        Resources/uTCN-300_4-10-13_release.json
        Resources/uTCN-300_4-10-13_release.bin
        Resources/mamba2_mag_mask_release.json
        Resources/mamba2_mag_mask_release.bin
        Resources/mamba2_mag_phase_mask_release.json
        Resources/mamba2_mag_phase_mask_release.bin
        Resources/mamba2_mag_phase_mask_release_d_128.json
        Resources/mamba2_mag_phase_mask_release_d_128.bin
        Resources/mamba2_mag_phase_mask_release_d_64_depth_1_hd16.json
        Resources/mamba2_mag_phase_mask_release_d_64_depth_1_hd16.bin
)

function(add_raw_cpu_benchmark TARGET PRODUCT_NAME PLUGIN_CODE MODEL_KIND JSON_FILE BIN_FILE)
    juce_add_plugin(${TARGET}
        COMPANY_NAME "SpectralBench"
        PLUGIN_MANUFACTURER_CODE SbcB
        PLUGIN_CODE ${PLUGIN_CODE}
        FORMATS AU VST3
        PRODUCT_NAME "${PRODUCT_NAME}")

    target_sources(${TARGET} PRIVATE
        Source/CpuBenchmarkProcessor.cpp
        Source/CpuBenchmarkProcessor.h
        Source/LstmRawModel.h
        Source/S6RawModel.h
        Source/TcnRawModel.h
        Source/RawModelUtils.h
        Source/TensorArchive.h
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
        CPU_BENCH_MODEL_KIND=${MODEL_KIND}
        CPU_BENCH_ORIGINAL_JSON="${JSON_FILE}"
        CPU_BENCH_ORIGINAL_BIN="${BIN_FILE}"
    )

    target_link_libraries(${TARGET} PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_processors
        juce::juce_audio_basics
        juce::juce_core
        CPUModelBenchmarksData
    )
endfunction()

add_raw_cpu_benchmark(LSTMRaw32CPUBenchmark "LSTM Raw 32 CPU Benchmark" Ls32 1
    "lstm_raw_32_release.json" "lstm_raw_32_release.bin")

add_raw_cpu_benchmark(S616DCPUBenchmark "S6 16D CPU Benchmark" S616 2
    "s6_16d_release.json" "s6_16d_release.bin")

add_raw_cpu_benchmark(UTCN100CPUBenchmark "uTCN 100 CPU Benchmark" T105 3
    "uTCN-100_4-10-5_release.json" "uTCN-100_4-10-5_release.bin")

add_raw_cpu_benchmark(UTCN300CPUBenchmark "uTCN 300 CPU Benchmark" T313 4
    "uTCN-300_4-10-13_release.json" "uTCN-300_4-10-13_release.bin")

function(add_mamba2_spectral_cpu_benchmark TARGET PRODUCT_NAME PLUGIN_CODE JSON_FILE BIN_FILE)
    juce_add_plugin(${TARGET}
        COMPANY_NAME "SpectralBench"
        PLUGIN_MANUFACTURER_CODE SbcB
        PLUGIN_CODE ${PLUGIN_CODE}
        FORMATS AU VST3
        PRODUCT_NAME "${PRODUCT_NAME}")

    target_sources(${TARGET} PRIVATE
        Source/Mamba2SpectralBenchmarkProcessor.cpp
        Source/Mamba2SpectralBenchmarkProcessor.h
        Source/Mamba2SpectralRawModel.h
        Source/RawModelUtils.h
        Source/TensorArchive.h
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
        CPU_BENCH_ORIGINAL_JSON="${JSON_FILE}"
        CPU_BENCH_ORIGINAL_BIN="${BIN_FILE}"
    )

    target_link_libraries(${TARGET} PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_processors
        juce::juce_audio_basics
        juce::juce_dsp
        juce::juce_core
        CPUModelBenchmarksData
    )
endfunction()

add_mamba2_spectral_cpu_benchmark(Mamba2MagMaskCPUBenchmark "Mamba2 Mag Mask CPU Benchmark" M2g1
    "mamba2_mag_mask_release.json" "mamba2_mag_mask_release.bin")

add_mamba2_spectral_cpu_benchmark(Mamba2MagPhaseMaskCPUBenchmark "Mamba2 Mag Phase Mask CPU Benchmark" M2p1
    "mamba2_mag_phase_mask_release.json" "mamba2_mag_phase_mask_release.bin")

function(add_mamba2_sized_spectral_cpu_benchmark TARGET PRODUCT_NAME PLUGIN_CODE VARIANT JSON_FILE BIN_FILE)
    juce_add_plugin(${TARGET}
        COMPANY_NAME "SpectralBench"
        PLUGIN_MANUFACTURER_CODE SbcB
        PLUGIN_CODE ${PLUGIN_CODE}
        FORMATS AU VST3
        PRODUCT_NAME "${PRODUCT_NAME}")

    target_sources(${TARGET} PRIVATE
        Source/Mamba2SizedSpectralBenchmarkProcessor.cpp
        Source/Mamba2SizedSpectralBenchmarkProcessor.h
        Source/Mamba2SizedSpectralRawModel.h
        Source/RawModelUtils.h
        Source/TensorArchive.h
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
        CPU_BENCH_SIZED_MAMBA_VARIANT=${VARIANT}
        CPU_BENCH_ORIGINAL_JSON="${JSON_FILE}"
        CPU_BENCH_ORIGINAL_BIN="${BIN_FILE}"
    )

    target_link_libraries(${TARGET} PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_processors
        juce::juce_audio_basics
        juce::juce_dsp
        juce::juce_core
        CPUModelBenchmarksData
    )
endfunction()

add_mamba2_sized_spectral_cpu_benchmark(SDRCMambaXSCPUBenchmark "SDRC XS CPU Benchmark" Sx12 128
    "mamba2_mag_phase_mask_release_d_128.json" "mamba2_mag_phase_mask_release_d_128.bin")

add_mamba2_sized_spectral_cpu_benchmark(SDRCMambaXXSCPUBenchmark "SDRC XXS CPU Benchmark" Sx64 64
    "mamba2_mag_phase_mask_release_d_64_depth_1_hd16.json" "mamba2_mag_phase_mask_release_d_64_depth_1_hd16.bin")

juce_add_plugin(S4C32F4CPUBenchmark
    COMPANY_NAME "SpectralBench"
    PLUGIN_MANUFACTURER_CODE SbcB
    PLUGIN_CODE S4b1
    FORMATS AU VST3
    PRODUCT_NAME "S4 C32 F4 CPU Benchmark")

target_sources(S4C32F4CPUBenchmark PRIVATE
    Source/S4BenchmarkProcessor.cpp
    Source/S4BenchmarkProcessor.h
    Source/S4RawModel.h
    Source/TensorArchive.h
)

target_compile_definitions(S4C32F4CPUBenchmark PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_USE_CAMERA=0
    JUCE_USE_MP3AUDIOFORMAT=0
    JUCE_USE_OGGVORBIS=0
    JUCE_USE_FLAC=0
    JUCE_USE_LAME_AUDIO_FORMAT=0
    JUCE_VST2_PLUGIN_ENABLE=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(S4C32F4CPUBenchmark PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_audio_basics
    juce::juce_core
    CPUModelBenchmarksData
)

add_executable(tcn_bench
    Source/tcn_bench.cpp
    Source/TcnRawModel.h
    Source/RawModelUtils.h
    Source/TensorArchive.h
)

target_link_libraries(tcn_bench PRIVATE
    juce::juce_core
)

target_compile_definitions(tcn_bench PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
)

add_executable(tcn_validate
    Source/tcn_validate.cpp
    Source/TcnRawModel.h
    Source/RawModelUtils.h
    Source/TensorArchive.h
)

target_link_libraries(tcn_validate PRIVATE
    juce::juce_core
)

target_compile_definitions(tcn_validate PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
)

add_executable(plugin_host_bench
    Source/plugin_host_bench.cpp
)

target_link_libraries(plugin_host_bench PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_audio_basics
    juce::juce_events
    juce::juce_core
)

target_compile_definitions(plugin_host_bench PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_PLUGINHOST_VST3=1
    JUCE_PLUGINHOST_AU=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_VST2_PLUGIN_ENABLE=0
    JUCE_MODAL_LOOPS_PERMITTED=1
)

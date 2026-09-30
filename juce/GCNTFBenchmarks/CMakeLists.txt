cmake_minimum_required(VERSION 3.22)
project(GCNTFBenchmarks VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(JUCE_DIR "" CACHE PATH "JUCE source directory")
set(MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../benchmarks/gcntf/results" CACHE PATH "Exported TorchScript models")
add_subdirectory("${JUCE_DIR}" JUCE)
find_package(Torch REQUIRED)
function(add_gcntf target product code stem strategy)
    juce_add_binary_data(${target}Data NAMESPACE ${target}Data HEADER_NAME ${target}Data.h
        SOURCES "${MODEL_DIR}/${stem}_${strategy}.pt")
    juce_add_plugin(${target} COMPANY_NAME SpectralBench
        PLUGIN_MANUFACTURER_CODE SbcB PLUGIN_CODE ${code}
        FORMATS VST3 PRODUCT_NAME "${product}")
    target_sources(${target} PRIVATE Source/Processor.cpp)
    target_compile_definitions(${target} PRIVATE
        MODEL_HEADER="${target}Data.h" MODEL_NAMESPACE=${target}Data
        JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0)
    target_link_libraries(${target} PRIVATE juce::juce_audio_utils ${target}Data ${TORCH_LIBRARIES})
    target_compile_options(${target} PRIVATE ${TORCH_CXX_FLAGS})
    set_target_properties(${target} PROPERTIES BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
endfunction()
add_gcntf(GCNTF250 "GCNTF 250 CPU benchmark" Gc25 gcntf_250_release conv)
add_gcntf(GCNTF2500Extended "GCNTF 2500 Extended CPU benchmark" Gc2e gcntf_2500_extended_release gather)
add_executable(gcntf_host Source/host.cpp)
target_link_libraries(gcntf_host PRIVATE juce::juce_audio_utils)
target_compile_definitions(gcntf_host PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0
    JUCE_PLUGINHOST_VST3=1 JUCE_PLUGINHOST_AU=0 JUCE_VST3_CAN_REPLACE_VST2=0 JUCE_MODAL_LOOPS_PERMITTED=1)

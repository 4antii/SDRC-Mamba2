#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include "TensorArchive.h"

class CpuBenchmarkAudioProcessor final : public juce::AudioProcessor
{
public:
    CpuBenchmarkAudioProcessor();

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    bool hasEditor() const override { return true; }
    juce::AudioProcessorEditor* createEditor() override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

private:
    juce::AudioProcessorValueTreeState params_;
    TensorArchive archive_;
    bool modelLoaded_ = false;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    juce::AudioBuffer<float> dry_;
    std::vector<float> tempIn_;
    std::vector<float> tempOut_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CpuBenchmarkAudioProcessor)
};

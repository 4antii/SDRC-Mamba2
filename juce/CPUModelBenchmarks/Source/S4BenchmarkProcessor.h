#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include "S4RawModel.h"
#include "TensorArchive.h"

class S4BenchmarkAudioProcessor final : public juce::AudioProcessor
{
public:
    S4BenchmarkAudioProcessor();

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

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
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& parameters() { return params_; }

private:
    juce::AudioProcessorValueTreeState params_;
    TensorArchive archive_;
    bool modelLoaded_ = false;
    std::vector<S4RawModel> models_;
    juce::AudioBuffer<float> dry_;
    std::vector<float> tempIn_;
    std::vector<float> tempOut_;
};

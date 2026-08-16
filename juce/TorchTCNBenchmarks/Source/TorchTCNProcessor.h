#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include <torch/script.h>
#include <torch/torch.h>

class TorchTCNAudioProcessor final : public juce::AudioProcessor
{
public:
    TorchTCNAudioProcessor();
    ~TorchTCNAudioProcessor() override = default;

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
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();
    bool loadModelFromBinaryData();
    void resizeBuffers(int samplesPerBlock, int numChannels);

    juce::AudioProcessorValueTreeState params_;
    torch::jit::script::Module model_;
    bool modelLoaded_ = false;
    int contextSamples_ = 0;
    int processSamples_ = 0;
    int maxBlockSamples_ = 0;

    std::vector<std::vector<float>> memory_;
    std::vector<std::vector<float>> processBuffers_;
    std::vector<std::vector<float>> outputBuffers_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TorchTCNAudioProcessor)
};

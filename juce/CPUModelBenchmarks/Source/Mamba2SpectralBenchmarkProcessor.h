#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

#include "Mamba2SpectralRawModel.h"
#include "TensorArchive.h"

class Mamba2SpectralBenchmarkAudioProcessor final : public juce::AudioProcessor
{
public:
    Mamba2SpectralBenchmarkAudioProcessor();

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
    struct ChannelState;

    void prepareChannels(int numChannels);
    void processHop(ChannelState& state, const float* params);
    static int getOrder(int nFft);

    juce::AudioProcessorValueTreeState params_;
    TensorArchive archive_;
    bool modelLoaded_ = false;
    int nFft_ = 512;
    int hop_ = 256;
    int freqBins_ = 257;
    std::unique_ptr<juce::dsp::FFT> fft_;
    std::vector<float> window_;
    std::vector<float> windowSq_;
    std::vector<ChannelState> channels_;
    juce::AudioBuffer<float> dry_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Mamba2SpectralBenchmarkAudioProcessor)
};

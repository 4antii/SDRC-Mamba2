#include "S4BenchmarkProcessor.h"

#include "BinaryData.h"

namespace
{
juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> p;
    juce::NormalisableRange<float> ratioRange { 1.0f, 100.0f };
    ratioRange.setSkewForCentre(10.0f);
    p.push_back(std::make_unique<juce::AudioParameterFloat>("threshold", "Threshold", -40.0f, 0.0f, -18.0f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("ratio", "Ratio", ratioRange, 4.0f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("attack", "Attack", 0.1f, 200.0f, 10.0f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("release", "Release", 50.0f, 3000.0f, 500.0f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("gain_db", "Gain", -24.0f, 24.0f, 0.0f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("mix", "Mix", 0.0f, 100.0f, 100.0f));
    return { p.begin(), p.end() };
}

const void* namedResource(const char* name, int& size)
{
    return BinaryData::getNamedResource(name, size);
}
}

S4BenchmarkAudioProcessor::S4BenchmarkAudioProcessor()
    : juce::AudioProcessor(BusesProperties()
                           .withInput("Input", juce::AudioChannelSet::stereo(), true)
                           .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      params_(*this, nullptr, "PARAMS", createLayout())
{
    int jsonSize = 0;
    int binSize = 0;
    const auto* json = namedResource("s4_c32_f4_release_json", jsonSize);
    const auto* bin = namedResource("s4_c32_f4_release_bin", binSize);
    modelLoaded_ = json != nullptr && bin != nullptr
        && archive_.load(json, static_cast<size_t>(jsonSize), bin, static_cast<size_t>(binSize));
}

void S4BenchmarkAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(sampleRate);
    models_.clear();
    models_.resize(static_cast<size_t>(juce::jmax(1, getTotalNumInputChannels())));
    if (modelLoaded_)
        for (auto& m : models_)
            m.load(archive_);
    dry_.setSize(getTotalNumInputChannels(), samplesPerBlock, false, false, true);
    tempIn_.resize(static_cast<size_t>(samplesPerBlock));
    tempOut_.resize(static_cast<size_t>(samplesPerBlock));
    setLatencySamples(0);
}

bool S4BenchmarkAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    if (layouts.getMainInputChannelSet() != layouts.getMainOutputChannelSet())
        return false;
    return layouts.getMainInputChannelSet() == juce::AudioChannelSet::mono()
        || layouts.getMainInputChannelSet() == juce::AudioChannelSet::stereo();
}

void S4BenchmarkAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused(midiMessages);
    juce::ScopedNoDenormals noDenormals;
    const int channels = getTotalNumInputChannels();
    const int samples = buffer.getNumSamples();
    for (int ch = channels; ch < getTotalNumOutputChannels(); ++ch)
        buffer.clear(ch, 0, samples);
    if (!modelLoaded_)
        return;

    if (dry_.getNumSamples() < samples)
        dry_.setSize(channels, samples, false, false, true);
    if (static_cast<int>(tempIn_.size()) < samples)
    {
        tempIn_.resize(static_cast<size_t>(samples));
        tempOut_.resize(static_cast<size_t>(samples));
    }

    const float paramNorm[4] = {
        params_.getRawParameterValue("threshold")->load() / 100.0f,
        params_.getRawParameterValue("ratio")->load() / 10.0f,
        params_.getRawParameterValue("attack")->load() / 1000.0f,
        params_.getRawParameterValue("release")->load() / 1000.0f
    };
    const float gain = juce::Decibels::decibelsToGain(params_.getRawParameterValue("gain_db")->load());
    const float mix = juce::jlimit(0.0f, 1.0f, params_.getRawParameterValue("mix")->load() / 100.0f);

    for (int ch = 0; ch < channels; ++ch)
    {
        dry_.copyFrom(ch, 0, buffer, ch, 0, samples);
        std::copy(buffer.getReadPointer(ch), buffer.getReadPointer(ch) + samples, tempIn_.begin());
        models_[static_cast<size_t>(ch)].processBlock(tempIn_.data(), tempOut_.data(), samples, paramNorm);
        auto* out = buffer.getWritePointer(ch);
        const auto* dry = dry_.getReadPointer(ch);
        for (int n = 0; n < samples; ++n)
            out[n] = (tempOut_[static_cast<size_t>(n)] * gain * mix) + (dry[n] * (1.0f - mix));
    }
}

juce::AudioProcessorEditor* S4BenchmarkAudioProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor(*this);
}

void S4BenchmarkAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    if (auto xml = params_.copyState().createXml())
        copyXmlToBinary(*xml, destData);
}

void S4BenchmarkAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml && xml->hasTagName(params_.state.getType()))
        params_.replaceState(juce::ValueTree::fromXml(*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new S4BenchmarkAudioProcessor();
}

#include "CpuBenchmarkProcessor.h"

#include "BinaryData.h"
#include "LstmRawModel.h"
#include "S6RawModel.h"
#include "TcnRawModel.h"

#include <cstring>

#ifndef CPU_BENCH_MODEL_KIND
#error CPU_BENCH_MODEL_KIND must be defined
#endif
#ifndef CPU_BENCH_ORIGINAL_JSON
#error CPU_BENCH_ORIGINAL_JSON must be defined
#endif
#ifndef CPU_BENCH_ORIGINAL_BIN
#error CPU_BENCH_ORIGINAL_BIN must be defined
#endif

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

const void* namedResourceForOriginalFilename(const char* originalFilename, int& size)
{
    for (int i = 0; i < BinaryData::namedResourceListSize; ++i)
    {
        const char* original = BinaryData::originalFilenames[i];
        if (original != nullptr && std::strcmp(original, originalFilename) == 0)
            return BinaryData::getNamedResource(BinaryData::namedResourceList[i], size);
    }
    size = 0;
    return nullptr;
}
}

struct CpuBenchmarkAudioProcessor::Impl
{
#if CPU_BENCH_MODEL_KIND == 1
    using Model = LstmRawModel;
#elif CPU_BENCH_MODEL_KIND == 2
    using Model = S6RawModel;
#elif CPU_BENCH_MODEL_KIND == 3 || CPU_BENCH_MODEL_KIND == 4
    using Model = TcnRawModel;
#else
#error Unknown CPU_BENCH_MODEL_KIND
#endif

    std::vector<Model> models;

    void load(const TensorArchive& archive, int channels)
    {
        models.clear();
        models.resize(static_cast<size_t>(juce::jmax(1, channels)));
        for (auto& model : models)
            model.load(archive);
    }

    void reset()
    {
        for (auto& model : models)
            model.reset();
    }

    void process(int channel, const float* input, float* output, int samples, const float* params)
    {
        models[static_cast<size_t>(channel)].processBlock(input, output, samples, params);
    }
};

CpuBenchmarkAudioProcessor::CpuBenchmarkAudioProcessor()
    : juce::AudioProcessor(BusesProperties()
                           .withInput("Input", juce::AudioChannelSet::stereo(), true)
                           .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      params_(*this, nullptr, "PARAMS", createLayout()),
      impl_(std::make_unique<Impl>())
{
    int jsonSize = 0;
    int binSize = 0;
    const auto* json = namedResourceForOriginalFilename(CPU_BENCH_ORIGINAL_JSON, jsonSize);
    const auto* bin = namedResourceForOriginalFilename(CPU_BENCH_ORIGINAL_BIN, binSize);
    modelLoaded_ = json != nullptr && bin != nullptr
        && archive_.load(json, static_cast<size_t>(jsonSize), bin, static_cast<size_t>(binSize));
}

void CpuBenchmarkAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(sampleRate);
    if (modelLoaded_)
        impl_->load(archive_, getTotalNumInputChannels());
    dry_.setSize(getTotalNumInputChannels(), samplesPerBlock, false, false, true);
    tempIn_.resize(static_cast<size_t>(samplesPerBlock));
    tempOut_.resize(static_cast<size_t>(samplesPerBlock));
    setLatencySamples(0);
}

bool CpuBenchmarkAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    if (layouts.getMainInputChannelSet() != layouts.getMainOutputChannelSet())
        return false;
    return layouts.getMainInputChannelSet() == juce::AudioChannelSet::mono()
        || layouts.getMainInputChannelSet() == juce::AudioChannelSet::stereo();
}

void CpuBenchmarkAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
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
        impl_->process(ch, tempIn_.data(), tempOut_.data(), samples, paramNorm);
        auto* out = buffer.getWritePointer(ch);
        const auto* dry = dry_.getReadPointer(ch);
        for (int n = 0; n < samples; ++n)
            out[n] = (tempOut_[static_cast<size_t>(n)] * gain * mix) + (dry[n] * (1.0f - mix));
    }
}

juce::AudioProcessorEditor* CpuBenchmarkAudioProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor(*this);
}

void CpuBenchmarkAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    if (auto xml = params_.copyState().createXml())
        copyXmlToBinary(*xml, destData);
}

void CpuBenchmarkAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml && xml->hasTagName(params_.state.getType()))
        params_.replaceState(juce::ValueTree::fromXml(*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new CpuBenchmarkAudioProcessor();
}

#include "TorchTCNProcessor.h"

#include "BinaryData.h"

#include <cstring>
#include <sstream>
#include <mutex>

#ifndef TORCH_TCN_MODEL_ORIGINAL
#error TORCH_TCN_MODEL_ORIGINAL must be defined
#endif
#ifndef TORCH_TCN_CONTEXT_SAMPLES
#error TORCH_TCN_CONTEXT_SAMPLES must be defined
#endif

namespace
{
std::once_flag torchThreadConfigFlag;

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

juce::AudioProcessorValueTreeState::ParameterLayout TorchTCNAudioProcessor::createLayout()
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

TorchTCNAudioProcessor::TorchTCNAudioProcessor()
    : juce::AudioProcessor(BusesProperties()
                           .withInput("Input", juce::AudioChannelSet::stereo(), true)
                           .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      params_(*this, nullptr, "PARAMS", createLayout()),
      contextSamples_(TORCH_TCN_CONTEXT_SAMPLES)
{
    std::call_once(torchThreadConfigFlag, [] {
        torch::set_num_threads(1);
        try
        {
            torch::set_num_interop_threads(1);
        }
        catch (const c10::Error&)
        {
        }
    });
    modelLoaded_ = loadModelFromBinaryData();
}

bool TorchTCNAudioProcessor::loadModelFromBinaryData()
{
    int modelSize = 0;
    const auto* data = namedResourceForOriginalFilename(TORCH_TCN_MODEL_ORIGINAL, modelSize);
    if (data == nullptr || modelSize <= 0)
        return false;

    try
    {
        std::string bytes(static_cast<const char*>(data), static_cast<size_t>(modelSize));
        std::istringstream stream(std::move(bytes), std::ios::binary);
        model_ = torch::jit::load(stream, torch::kCPU);
        model_.eval();
        return true;
    }
    catch (const c10::Error&)
    {
        return false;
    }
}

void TorchTCNAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(sampleRate);
    resizeBuffers(samplesPerBlock, getTotalNumInputChannels());
    setLatencySamples(0);
}

void TorchTCNAudioProcessor::resizeBuffers(int samplesPerBlock, int numChannels)
{
    maxBlockSamples_ = juce::jmax(1, samplesPerBlock);
    processSamples_ = contextSamples_ + maxBlockSamples_;
    const int channels = juce::jmax(1, numChannels);
    memory_.assign(static_cast<size_t>(channels), std::vector<float>(static_cast<size_t>(contextSamples_), 0.0f));
    processBuffers_.assign(static_cast<size_t>(channels), std::vector<float>(static_cast<size_t>(processSamples_), 0.0f));
    outputBuffers_.assign(static_cast<size_t>(channels), std::vector<float>(static_cast<size_t>(maxBlockSamples_), 0.0f));
}

bool TorchTCNAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    if (layouts.getMainInputChannelSet() != layouts.getMainOutputChannelSet())
        return false;
    return layouts.getMainInputChannelSet() == juce::AudioChannelSet::mono()
        || layouts.getMainInputChannelSet() == juce::AudioChannelSet::stereo();
}

void TorchTCNAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused(midiMessages);
    juce::ScopedNoDenormals noDenormals;

    const int numSamples = buffer.getNumSamples();
    const int channels = getTotalNumInputChannels();
    for (int ch = channels; ch < getTotalNumOutputChannels(); ++ch)
        buffer.clear(ch, 0, numSamples);

    if (!modelLoaded_)
        return;
    if (numSamples > maxBlockSamples_ || static_cast<int>(memory_.size()) < channels)
        resizeBuffers(numSamples, channels);

    const float params[4] = {
        params_.getRawParameterValue("threshold")->load() / 100.0f,
        params_.getRawParameterValue("ratio")->load() / 10.0f,
        params_.getRawParameterValue("attack")->load() / 1000.0f,
        params_.getRawParameterValue("release")->load() / 1000.0f
    };
    const float gain = juce::Decibels::decibelsToGain(params_.getRawParameterValue("gain_db")->load());
    const float mix = juce::jlimit(0.0f, 1.0f, params_.getRawParameterValue("mix")->load() / 100.0f);

    try
    {
        torch::NoGradGuard noGrad;
        auto p = torch::from_blob(const_cast<float*>(params), {1, 1, 4}, torch::kFloat32).clone();

        for (int ch = 0; ch < channels; ++ch)
        {
            auto& memory = memory_[static_cast<size_t>(ch)];
            auto& processBuffer = processBuffers_[static_cast<size_t>(ch)];
            auto& outputBuffer = outputBuffers_[static_cast<size_t>(ch)];
            std::copy(memory.begin(), memory.end(), processBuffer.begin());
            const auto* input = buffer.getReadPointer(ch);
            std::copy(input, input + numSamples, processBuffer.begin() + contextSamples_);
            if (contextSamples_ > 0)
                std::copy(processBuffer.begin() + numSamples,
                          processBuffer.begin() + numSamples + contextSamples_,
                          memory.begin());

            auto frame = torch::from_blob(processBuffer.data(), {1, 1, contextSamples_ + numSamples}, torch::kFloat32);
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(frame);
            inputs.emplace_back(p);
            auto outTensor = model_.forward(inputs).toTensor().contiguous();
            const auto* out = outTensor.data_ptr<float>();
            const int outSamples = static_cast<int>(outTensor.size(2));
            const int copySamples = juce::jmin(numSamples, outSamples);
            std::copy(out + (outSamples - copySamples), out + outSamples, outputBuffer.begin());
        }
    }
    catch (const c10::Error&)
    {
        return;
    }

    for (int ch = 0; ch < channels; ++ch)
    {
        auto* dst = buffer.getWritePointer(ch);
        const auto* dry = buffer.getReadPointer(ch);
        const auto& outputBuffer = outputBuffers_[static_cast<size_t>(ch)];
        for (int n = 0; n < numSamples; ++n)
        {
            const float wet = outputBuffer[static_cast<size_t>(n)] * gain;
            dst[n] = wet * mix + dry[n] * (1.0f - mix);
        }
    }
}

juce::AudioProcessorEditor* TorchTCNAudioProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor(*this);
}

void TorchTCNAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    if (auto xml = params_.copyState().createXml())
        copyXmlToBinary(*xml, destData);
}

void TorchTCNAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml && xml->hasTagName(params_.state.getType()))
        params_.replaceState(juce::ValueTree::fromXml(*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new TorchTCNAudioProcessor();
}

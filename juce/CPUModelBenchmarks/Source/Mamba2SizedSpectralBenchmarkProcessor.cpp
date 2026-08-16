#include "Mamba2SizedSpectralBenchmarkProcessor.h"

#include "BinaryData.h"

#include <array>
#include <cstring>

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

struct Mamba2SizedSpectralBenchmarkAudioProcessor::ChannelState
{
    BenchmarkSizedMambaModel model;
    std::vector<float> inputBuf;
    std::vector<float> outputBuf;
    std::vector<float> windowSum;
    std::vector<float> outQueue;
    std::vector<juce::dsp::Complex<float>> fftIn;
    std::vector<juce::dsp::Complex<float>> fftOut;
    std::array<float, BenchmarkSizedMambaModel::kFreqBins> magLin {};
    std::array<float, BenchmarkSizedMambaModel::kFreqBins> phase {};
    std::array<float, BenchmarkSizedMambaModel::kFreqBins> cosPhase {};
    std::array<float, BenchmarkSizedMambaModel::kFreqBins> sinPhase {};
    std::array<float, BenchmarkSizedMambaModel::kFreqBins> mask {};
    std::array<float, BenchmarkSizedMambaModel::kFreqBins> dphi {};
    int hopFill = 0;
    int outRead = 0;
    int inputWrite = 0;
};

Mamba2SizedSpectralBenchmarkAudioProcessor::Mamba2SizedSpectralBenchmarkAudioProcessor()
    : juce::AudioProcessor(BusesProperties()
                           .withInput("Input", juce::AudioChannelSet::stereo(), true)
                           .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      params_(*this, nullptr, "PARAMS", createLayout())
{
    int jsonSize = 0;
    int binSize = 0;
    const auto* json = namedResourceForOriginalFilename(CPU_BENCH_ORIGINAL_JSON, jsonSize);
    const auto* bin = namedResourceForOriginalFilename(CPU_BENCH_ORIGINAL_BIN, binSize);
    modelLoaded_ = json != nullptr && bin != nullptr
        && archive_.load(json, static_cast<size_t>(jsonSize), bin, static_cast<size_t>(binSize));
    if (modelLoaded_)
    {
        nFft_ = archive_.getConfigInt("n_fft", 512);
        hop_ = archive_.getConfigInt("hop_length", 256);
        freqBins_ = nFft_ / 2 + 1;
    }
}

void Mamba2SizedSpectralBenchmarkAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(sampleRate);
    fft_ = std::make_unique<juce::dsp::FFT>(getOrder(nFft_));
    window_.resize(static_cast<size_t>(nFft_));
    windowSq_.resize(static_cast<size_t>(nFft_));
    const float twoPi = 2.0f * juce::MathConstants<float>::pi;
    for (int i = 0; i < nFft_; ++i)
    {
        const float phase = twoPi * static_cast<float>(i) / static_cast<float>(nFft_);
        const float w = 0.54f - 0.46f * std::cos(phase);
        window_[static_cast<size_t>(i)] = w;
        windowSq_[static_cast<size_t>(i)] = w * w;
    }
    if (modelLoaded_)
        prepareChannels(getTotalNumInputChannels());
    dry_.setSize(getTotalNumInputChannels(), samplesPerBlock, false, false, true);
    setLatencySamples(nFft_);
}

void Mamba2SizedSpectralBenchmarkAudioProcessor::prepareChannels(int numChannels)
{
    channels_.clear();
    channels_.resize(static_cast<size_t>(juce::jmax(1, numChannels)));
    for (auto& ch : channels_)
    {
        ch.model.load(archive_);
        ch.inputBuf.assign(static_cast<size_t>(nFft_), 0.0f);
        ch.outputBuf.assign(static_cast<size_t>(nFft_), 0.0f);
        ch.windowSum.assign(static_cast<size_t>(nFft_), 0.0f);
        ch.outQueue.assign(static_cast<size_t>(hop_), 0.0f);
        ch.fftIn.assign(static_cast<size_t>(nFft_), {});
        ch.fftOut.assign(static_cast<size_t>(nFft_), {});
        ch.hopFill = 0;
        ch.outRead = 0;
        ch.inputWrite = 0;
    }
}

bool Mamba2SizedSpectralBenchmarkAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    if (layouts.getMainInputChannelSet() != layouts.getMainOutputChannelSet())
        return false;
    return layouts.getMainInputChannelSet() == juce::AudioChannelSet::mono()
        || layouts.getMainInputChannelSet() == juce::AudioChannelSet::stereo();
}

void Mamba2SizedSpectralBenchmarkAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
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
        auto* data = buffer.getWritePointer(ch);
        auto& state = channels_[static_cast<size_t>(ch)];
        const auto* dry = dry_.getReadPointer(ch);

        for (int n = 0; n < samples; ++n)
        {
            state.inputBuf[static_cast<size_t>(state.inputWrite)] = dry[n];
            if (++state.inputWrite >= nFft_)
                state.inputWrite = 0;
            ++state.hopFill;

            if (state.hopFill == hop_)
            {
                processHop(state, paramNorm);
                state.hopFill = 0;
                state.outRead = 0;
            }

            float wet = 0.0f;
            if (state.outRead < hop_)
            {
                wet = state.outQueue[static_cast<size_t>(state.outRead)] * gain;
                ++state.outRead;
            }
            data[n] = wet * mix + dry[n] * (1.0f - mix);
        }
    }
}

void Mamba2SizedSpectralBenchmarkAudioProcessor::processHop(ChannelState& state, const float* params)
{
    const int frameStart = state.inputWrite;
    int idx = frameStart;
    for (int i = 0; i < nFft_; ++i)
    {
        const float x = state.inputBuf[static_cast<size_t>(idx)] * window_[static_cast<size_t>(i)];
        state.fftIn[static_cast<size_t>(i)] = { x, 0.0f };
        if (++idx >= nFft_)
            idx = 0;
    }

    fft_->perform(state.fftIn.data(), state.fftOut.data(), false);

    for (int k = 0; k < freqBins_; ++k)
    {
        const auto c = state.fftOut[static_cast<size_t>(k)];
        const float re = c.real();
        const float im = c.imag();
        const float mag = std::sqrt(re * re + im * im);
        const float ph = std::atan2(im, re);
        state.magLin[static_cast<size_t>(k)] = mag;
        state.phase[static_cast<size_t>(k)] = ph;
        state.cosPhase[static_cast<size_t>(k)] = std::cos(ph);
        state.sinPhase[static_cast<size_t>(k)] = std::sin(ph);
    }

    state.model.processFrame(state.magLin.data(), state.cosPhase.data(), state.sinPhase.data(),
                             params, state.mask.data(), state.dphi.data());

    for (int k = 0; k < freqBins_; ++k)
    {
        const float magHat = state.magLin[static_cast<size_t>(k)] * state.mask[static_cast<size_t>(k)];
        const float phiHat = state.phase[static_cast<size_t>(k)] + state.dphi[static_cast<size_t>(k)];
        state.fftOut[static_cast<size_t>(k)] = { magHat * std::cos(phiHat), magHat * std::sin(phiHat) };
        if (k > 0 && k < (nFft_ / 2))
            state.fftOut[static_cast<size_t>(nFft_ - k)] = std::conj(state.fftOut[static_cast<size_t>(k)]);
    }

    fft_->perform(state.fftOut.data(), state.fftIn.data(), true);

    idx = frameStart;
    for (int i = 0; i < nFft_; ++i)
    {
        const float sample = state.fftIn[static_cast<size_t>(i)].real() * window_[static_cast<size_t>(i)];
        state.outputBuf[static_cast<size_t>(idx)] += sample;
        state.windowSum[static_cast<size_t>(idx)] += windowSq_[static_cast<size_t>(i)];
        if (++idx >= nFft_)
            idx = 0;
    }

    idx = frameStart;
    for (int i = 0; i < hop_; ++i)
    {
        float y = 0.0f;
        const float denom = state.windowSum[static_cast<size_t>(idx)];
        if (denom > 0.0f)
            y = state.outputBuf[static_cast<size_t>(idx)] / denom;
        state.outQueue[static_cast<size_t>(i)] = y;
        state.outputBuf[static_cast<size_t>(idx)] = 0.0f;
        state.windowSum[static_cast<size_t>(idx)] = 0.0f;
        if (++idx >= nFft_)
            idx = 0;
    }
}

juce::AudioProcessorEditor* Mamba2SizedSpectralBenchmarkAudioProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor(*this);
}

void Mamba2SizedSpectralBenchmarkAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    if (auto xml = params_.copyState().createXml())
        copyXmlToBinary(*xml, destData);
}

void Mamba2SizedSpectralBenchmarkAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml && xml->hasTagName(params_.state.getType()))
        params_.replaceState(juce::ValueTree::fromXml(*xml));
}

int Mamba2SizedSpectralBenchmarkAudioProcessor::getOrder(int nFft)
{
    int order = 0;
    while ((1 << order) < nFft)
        ++order;
    return order;
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new Mamba2SizedSpectralBenchmarkAudioProcessor();
}

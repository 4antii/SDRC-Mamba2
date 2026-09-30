#include <juce_audio_utils/juce_audio_utils.h>
#include <torch/script.h>
#include <ATen/Parallel.h>
#include <sstream>
#include MODEL_HEADER

class GCNTFProcessor final : public juce::AudioProcessor
{
public:
    static constexpr int frameSize = 512;
    GCNTFProcessor() : AudioProcessor(BusesProperties().withInput("Input", juce::AudioChannelSet::stereo(), true)
                                     .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
        params(*this, nullptr, "PARAMS", layout())
    {
        at::set_num_threads(1);
        try { at::set_num_interop_threads(1); } catch (const c10::Error&) {}
        int size = 0;
        const auto* bytes = MODEL_NAMESPACE::getNamedResource(MODEL_NAMESPACE::namedResourceList[0], size);
        for (auto& model : models)
        {
            std::istringstream stream(std::string(bytes, size), std::ios::binary);
            model = torch::jit::load(stream, torch::kCPU);
            model.eval();
        }
        controls = torch::zeros({1, 4});
        for (int ch = 0; ch < 2; ++ch)
            frames[ch] = torch::from_blob(input[ch].data(), {1, 1, frameSize}, torch::kFloat32);
    }
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return frameSize / rate; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}
    bool hasEditor() const override { return true; }
    juce::AudioProcessorEditor* createEditor() override { return new juce::GenericAudioProcessorEditor(*this); }
    bool isBusesLayoutSupported(const BusesLayout& buses) const override
    {
        return buses.getMainInputChannelSet() == buses.getMainOutputChannelSet()
            && (buses.getMainInputChannelSet() == juce::AudioChannelSet::mono()
                || buses.getMainInputChannelSet() == juce::AudioChannelSet::stereo());
    }
    void prepareToPlay(double sampleRate, int) override
    {
        rate = sampleRate;
        // The trained frequency/time mapping is 44.1 kHz; no implicit resampling.
        at::NoGradGuard guard;
        for (auto& model : models) model.run_method("reset");
        for (auto& a : input) a.fill(0);
        for (auto& a : output) a.fill(0);
        for (auto& a : dry) a.fill(0);
        position = 0;
        setLatencySamples(frameSize);
    }
    void releaseResources() override {}
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
    {
        juce::ScopedNoDenormals noDenormals;
        at::NoGradGuard guard;
        auto* p = controls.data_ptr<float>();
        p[0] = params.getRawParameterValue("threshold")->load() / 100.f;
        p[1] = params.getRawParameterValue("ratio")->load() / 10.f;
        p[2] = params.getRawParameterValue("attack")->load() / 1000.f;
        p[3] = params.getRawParameterValue("release")->load() / 1000.f;
        const float gain = juce::Decibels::decibelsToGain(params.getRawParameterValue("gain_db")->load());
        const float mix = params.getRawParameterValue("mix")->load() / 100.f;
        const int channels = getTotalNumInputChannels();
        for (int n = 0; n < buffer.getNumSamples(); ++n)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                const float x = buffer.getSample(ch, n);
                input[ch][position] = x;
                buffer.setSample(ch, n, mix * gain * output[ch][position] + (1.f - mix) * dry[ch][position]);
                dry[ch][position] = x;
            }
            if (++position == frameSize)
            {
                for (int ch = 0; ch < channels; ++ch)
                {
                    auto result = models[ch].forward({frames[ch], controls}).toTensor().contiguous();
                    std::copy_n(result.data_ptr<float>(), frameSize, output[ch].begin());
                }
                position = 0;
            }
        }
    }
    void getStateInformation(juce::MemoryBlock& dest) override
    { if (auto xml = params.copyState().createXml()) copyXmlToBinary(*xml, dest); }
    void setStateInformation(const void* data, int size) override
    { auto xml = getXmlFromBinary(data, size); if (xml && xml->hasTagName("PARAMS")) params.replaceState(juce::ValueTree::fromXml(*xml)); }
private:
    static juce::AudioProcessorValueTreeState::ParameterLayout layout()
    {
        std::vector<std::unique_ptr<juce::RangedAudioParameter>> p;
        juce::NormalisableRange<float> ratio(1.f, 100.f); ratio.setSkewForCentre(10.f);
        p.push_back(std::make_unique<juce::AudioParameterFloat>("threshold", "Threshold", -40.f, 0.f, -18.f));
        p.push_back(std::make_unique<juce::AudioParameterFloat>("ratio", "Ratio", ratio, 4.f));
        p.push_back(std::make_unique<juce::AudioParameterFloat>("attack", "Attack", .1f, 200.f, 10.f));
        p.push_back(std::make_unique<juce::AudioParameterFloat>("release", "Release", 50.f, 3000.f, 500.f));
        p.push_back(std::make_unique<juce::AudioParameterFloat>("gain_db", "Gain", -24.f, 24.f, 0.f));
        p.push_back(std::make_unique<juce::AudioParameterFloat>("mix", "Mix", 0.f, 100.f, 100.f));
        return {p.begin(), p.end()};
    }
    juce::AudioProcessorValueTreeState params;
    std::array<torch::jit::Module, 2> models;
    std::array<torch::Tensor, 2> frames;
    torch::Tensor controls;
    std::array<std::array<float, frameSize>, 2> input{}, output{}, dry{};
    int position = 0;
    double rate = 44100;
};
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new GCNTFProcessor(); }

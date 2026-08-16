#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace
{
struct Options
{
    double sampleRate = 48000.0;
    int blockSize = 512;
    int channels = 2;
    int warmupBlocks = 50;
    int iterations = 500;
    juce::File pluginRoot { juce::File::getCurrentWorkingDirectory().getChildFile("build") };
    juce::File csvOut { juce::File::getCurrentWorkingDirectory().getChildFile("plugin_host_benchmark.csv") };
    std::vector<juce::File> pluginFiles;
};

struct Result
{
    juce::String name;
    juce::String path;
    juce::String status;
    double sampleRate = 0.0;
    int blockSize = 0;
    int channels = 0;
    double deadlineMs = 0.0;
    double meanMs = 0.0;
    double stdMs = 0.0;
    double p95Ms = 0.0;
    double p99Ms = 0.0;
    double maxMs = 0.0;
    double cpuPercent = 0.0;
    double rtf = 0.0;
    int misses = 0;
};

juce::String csvEscape(const juce::String& s)
{
    auto out = s.replace("\"", "\"\"");
    if (out.containsAnyOf(",\n\""))
        return "\"" + out + "\"";
    return out;
}

int percentileIndex(int n, double q)
{
    if (n <= 1)
        return 0;
    return juce::jlimit(0, n - 1, static_cast<int>(std::ceil(q * static_cast<double>(n)) - 1.0));
}

std::vector<juce::File> defaultPluginPaths(const juce::File& root)
{
    const char* rels[] = {
        "LSTMRaw32CPUBenchmark_artefacts/Release/VST3/LSTM Raw 32 CPU Benchmark.vst3",
        "UTCN100CPUBenchmark_artefacts/Release/VST3/uTCN 100 CPU Benchmark.vst3",
        "UTCN300CPUBenchmark_artefacts/Release/VST3/uTCN 300 CPU Benchmark.vst3",
        "S4C32F4CPUBenchmark_artefacts/Release/VST3/S4 C32 F4 CPU Benchmark.vst3",
        "S616DCPUBenchmark_artefacts/Release/VST3/S6 16D CPU Benchmark.vst3",
        "Mamba2MagMaskCPUBenchmark_artefacts/Release/VST3/Mamba2 Mag Mask CPU Benchmark.vst3",
        "Mamba2MagPhaseMaskCPUBenchmark_artefacts/Release/VST3/Mamba2 Mag Phase Mask CPU Benchmark.vst3",
        "SDRCMambaXSCPUBenchmark_artefacts/Release/VST3/SDRC XS CPU Benchmark.vst3",
        "SDRCMambaXXSCPUBenchmark_artefacts/Release/VST3/SDRC XXS CPU Benchmark.vst3",
    };

    std::vector<juce::File> files;
    for (auto* rel : rels)
        files.push_back(root.getChildFile(rel));
    return files;
}

void fillInput(juce::AudioBuffer<float>& buffer, int64_t startSample, double sampleRate)
{
    constexpr double twoPi = 6.2831853071795864769;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        auto* dst = buffer.getWritePointer(ch);
        const double freq = ch == 0 ? 997.0 : 1499.0;
        for (int n = 0; n < buffer.getNumSamples(); ++n)
        {
            const double t = static_cast<double>(startSample + n) / sampleRate;
            dst[n] = static_cast<float>(0.1 * std::sin(twoPi * freq * t)
                                      + 0.03 * std::sin(twoPi * 71.0 * t));
        }
    }
}

void setDefaultParams(juce::AudioPluginInstance& plugin)
{
    for (auto* p : plugin.getParameters())
    {
        const auto name = p->getName(128).toLowerCase();
        if (name.contains("threshold"))
            p->setValueNotifyingHost(0.55f);
        else if (name == "mix")
            p->setValueNotifyingHost(1.0f);
        else if (name.contains("gain"))
            p->setValueNotifyingHost(0.5f);
    }
}

Result benchmarkPlugin(const juce::File& file, const Options& opt)
{
    Result r;
    r.path = file.getFullPathName();
    r.sampleRate = opt.sampleRate;
    r.blockSize = opt.blockSize;
    r.channels = opt.channels;
    r.deadlineMs = static_cast<double>(opt.blockSize) / opt.sampleRate * 1000.0;

    if (!file.exists())
    {
        r.name = file.getFileNameWithoutExtension();
        r.status = "missing";
        return r;
    }

    juce::VST3PluginFormat format;
    juce::OwnedArray<juce::PluginDescription> types;
    format.findAllTypesForFile(types, file.getFullPathName());
    if (types.isEmpty())
    {
        r.name = file.getFileNameWithoutExtension();
        r.status = "scan_failed";
        return r;
    }

    juce::AudioPluginFormatManager manager;
    manager.addFormat(new juce::VST3PluginFormat());
    juce::String error;
    std::unique_ptr<juce::AudioPluginInstance> plugin;
    bool done = false;
    manager.createPluginInstanceAsync(*types[0], opt.sampleRate, opt.blockSize,
                                      [&] (std::unique_ptr<juce::AudioPluginInstance> instance, const juce::String& err)
                                      {
                                          plugin = std::move(instance);
                                          error = err;
                                          done = true;
                                      });
    while (!done)
        juce::MessageManager::getInstance()->runDispatchLoopUntil(10);

    if (plugin == nullptr)
    {
        r.name = types[0]->name;
        r.status = "load_failed: " + error;
        return r;
    }

    r.name = plugin->getName();
    const auto set = opt.channels == 1 ? juce::AudioChannelSet::mono() : juce::AudioChannelSet::stereo();
    juce::AudioProcessor::BusesLayout layout;
    layout.inputBuses.add(set);
    layout.outputBuses.add(set);
    plugin->setBusesLayout(layout);
    plugin->setRateAndBufferSizeDetails(opt.sampleRate, opt.blockSize);
    setDefaultParams(*plugin);
    plugin->prepareToPlay(opt.sampleRate, opt.blockSize);

    juce::AudioBuffer<float> buffer(opt.channels, opt.blockSize);
    juce::MidiBuffer midi;
    int64_t sample = 0;

    for (int i = 0; i < opt.warmupBlocks; ++i)
    {
        fillInput(buffer, sample, opt.sampleRate);
        plugin->processBlock(buffer, midi);
        sample += opt.blockSize;
        midi.clear();
    }

    std::vector<double> times;
    times.reserve(static_cast<size_t>(opt.iterations));
    for (int i = 0; i < opt.iterations; ++i)
    {
        fillInput(buffer, sample, opt.sampleRate);
        const auto t0 = std::chrono::high_resolution_clock::now();
        plugin->processBlock(buffer, midi);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
        if (ms > r.deadlineMs)
            ++r.misses;
        sample += opt.blockSize;
        midi.clear();
    }

    plugin->releaseResources();

    const double sum = std::accumulate(times.begin(), times.end(), 0.0);
    r.meanMs = sum / static_cast<double>(times.size());
    double sq = 0.0;
    for (double v : times)
        sq += (v - r.meanMs) * (v - r.meanMs);
    r.stdMs = std::sqrt(sq / static_cast<double>(times.size()));
    std::sort(times.begin(), times.end());
    r.p95Ms = times[static_cast<size_t>(percentileIndex(static_cast<int>(times.size()), 0.95))];
    r.p99Ms = times[static_cast<size_t>(percentileIndex(static_cast<int>(times.size()), 0.99))];
    r.maxMs = times.back();
    r.cpuPercent = 100.0 * r.meanMs / r.deadlineMs;
    r.rtf = r.deadlineMs / r.meanMs;
    r.status = "ok";
    return r;
}

void writeCsv(const juce::File& out, const std::vector<Result>& rows)
{
    out.getParentDirectory().createDirectory();
    std::ofstream f(out.getFullPathName().toStdString());
    f << "name,status,path,sample_rate,block_size,channels,deadline_ms,mean_ms,std_ms,p95_ms,p99_ms,max_ms,cpu_percent,rtf,deadline_misses\n";
    for (const auto& r : rows)
    {
        f << csvEscape(r.name) << ','
          << csvEscape(r.status) << ','
          << csvEscape(r.path) << ','
          << r.sampleRate << ','
          << r.blockSize << ','
          << r.channels << ','
          << r.deadlineMs << ','
          << r.meanMs << ','
          << r.stdMs << ','
          << r.p95Ms << ','
          << r.p99Ms << ','
          << r.maxMs << ','
          << r.cpuPercent << ','
          << r.rtf << ','
          << r.misses << '\n';
    }
}

Options parseArgs(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--sample-rate") opt.sampleRate = std::atof(next());
        else if (a == "--block-size") opt.blockSize = std::atoi(next());
        else if (a == "--channels") opt.channels = std::atoi(next());
        else if (a == "--warmup") opt.warmupBlocks = std::atoi(next());
        else if (a == "--iterations") opt.iterations = std::atoi(next());
        else if (a == "--plugin-root") opt.pluginRoot = juce::File(next());
        else if (a == "--csv") opt.csvOut = juce::File(next());
        else if (a == "--plugin") opt.pluginFiles.emplace_back(next());
    }
    opt.channels = opt.channels <= 1 ? 1 : 2;
    opt.blockSize = juce::jmax(1, opt.blockSize);
    opt.iterations = juce::jmax(1, opt.iterations);
    return opt;
}
}

int main(int argc, char** argv)
{
    juce::ScopedJuceInitialiser_GUI init;
    const auto opt = parseArgs(argc, argv);
    std::vector<Result> rows;

    const auto files = opt.pluginFiles.empty() ? defaultPluginPaths(opt.pluginRoot) : opt.pluginFiles;
    for (const auto& file : files)
    {
        auto r = benchmarkPlugin(file, opt);
        std::cout << r.name << " | " << r.status;
        if (r.status == "ok")
            std::cout << " | mean " << r.meanMs << " ms | CPU " << r.cpuPercent << "% | p99 " << r.p99Ms << " ms | misses " << r.misses;
        std::cout << std::endl;
        rows.push_back(std::move(r));
    }

    writeCsv(opt.csvOut, rows);
    std::cout << "CSV: " << opt.csvOut.getFullPathName() << std::endl;
    return 0;
}

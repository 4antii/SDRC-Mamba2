#include "TcnRawModel.h"
#include "TensorArchive.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "usage: tcn_bench model.json model.bin iterations [block]\n";
        return 2;
    }
    const int iterations = std::atoi(argv[3]);
    const int block = argc > 4 ? std::atoi(argv[4]) : 512;

    std::ifstream jf(argv[1], std::ios::binary);
    std::ifstream bf(argv[2], std::ios::binary);
    std::vector<char> json((std::istreambuf_iterator<char>(jf)), {});
    std::vector<char> bin((std::istreambuf_iterator<char>(bf)), {});
    TensorArchive archive;
    if (!archive.load(json.data(), json.size(), bin.data(), bin.size()))
    {
        std::cerr << "archive load failed\n";
        return 3;
    }

    TcnRawModel model;
    model.load(archive);
    std::vector<float> x(static_cast<size_t>(block));
    std::vector<float> y(static_cast<size_t>(block));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& v : x) v = dist(rng);
    const float params[4] = {0.25f, 0.5f, 0.5f, 0.5f};

    for (int i = 0; i < 3; ++i)
        model.processBlock(x.data(), y.data(), block, params);

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
        model.processBlock(x.data(), y.data(), block, params);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double ms = sec * 1000.0 / iterations;
    const double blockMs48k = 1000.0 * block / 48000.0;
    std::cout << "context_samples," << model.latencySamples() << "\n";
    std::cout << "block_samples," << block << "\n";
    std::cout << "iterations," << iterations << "\n";
    std::cout << "ms_per_block," << ms << "\n";
    std::cout << "block_budget_ms_48k," << blockMs48k << "\n";
    std::cout << "single_channel_cpu_percent_48k," << (ms / blockMs48k * 100.0) << "\n";
    std::cout << "last_sample," << y.back() << "\n";
    return 0;
}

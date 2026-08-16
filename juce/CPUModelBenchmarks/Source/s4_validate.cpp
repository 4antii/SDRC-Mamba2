#include <fstream>
#include <iostream>
#include <vector>

#include "S4RawModel.h"

static std::vector<char> readFile(const char* path)
{
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "usage: s4_validate model.json model.bin input.txt\n";
        return 2;
    }

    auto json = readFile(argv[1]);
    auto bin = readFile(argv[2]);
    if (json.empty() || bin.empty())
    {
        std::cerr << "failed to read archive files\n";
        return 1;
    }

    TensorArchive archive;
    if (!archive.load(json.data(), json.size(), bin.data(), bin.size()))
    {
        std::cerr << "failed to load TensorArchive\n";
        return 1;
    }

    std::ifstream inputFile(argv[3]);
    std::vector<float> input;
    float v = 0.0f;
    while (inputFile >> v)
        input.push_back(v);
    if (input.empty())
    {
        std::cerr << "empty input\n";
        return 1;
    }

    S4RawModel model;
    model.load(archive);
    std::vector<float> output(input.size(), 0.0f);
    const float params[4] = { 0.25f, 0.5f, 0.5f, 0.5f };
    model.processBlock(input.data(), output.data(), static_cast<int>(input.size()), params);

    std::cout.setf(std::ios::scientific);
    for (float y : output)
        std::cout << y << "\n";
    return 0;
}

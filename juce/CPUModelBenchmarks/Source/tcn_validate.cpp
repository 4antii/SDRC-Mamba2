#include "TcnRawModel.h"
#include "TensorArchive.h"

#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: tcn_validate model.json model.bin input.txt\n";
        return 2;
    }
    std::ifstream jf(argv[1], std::ios::binary);
    std::ifstream bf(argv[2], std::ios::binary);
    std::ifstream xf(argv[3]);
    std::vector<char> json((std::istreambuf_iterator<char>(jf)), {});
    std::vector<char> bin((std::istreambuf_iterator<char>(bf)), {});
    std::vector<float> input;
    for (float v = 0.0f; xf >> v;)
        input.push_back(v);
    TensorArchive archive;
    if (!archive.load(json.data(), json.size(), bin.data(), bin.size()))
        return 3;
    TcnRawModel model;
    model.load(archive);
    std::vector<float> output(input.size());
    const float params[4] = {0.25f, 0.5f, 0.5f, 0.5f};
    model.processBlock(input.data(), output.data(), static_cast<int>(input.size()), params);
    for (float y : output)
        std::cout << y << '\n';
    return 0;
}

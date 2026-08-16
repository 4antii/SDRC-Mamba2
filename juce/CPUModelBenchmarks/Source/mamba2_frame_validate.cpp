#include "Mamba2SizedSpectralRawModel.h"
#include "Mamba2SpectralRawModel.h"
#include "TensorArchive.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{
template <typename Model>
int runModel(const std::vector<char>& json, const std::vector<char>& bin, const char* framesPath)
{
    TensorArchive archive;
    if (!archive.load(json.data(), json.size(), bin.data(), bin.size()))
        return 4;

    std::ifstream ff(framesPath);
    if (!ff)
        return 5;

    Model model;
    model.load(archive);

    constexpr int bins = 257;
    std::vector<float> mag(bins), cosPhase(bins), sinPhase(bins), mask(bins), dphi(bins);
    const float params[4] = { 0.25f, 0.5f, 0.5f, 0.5f };

    std::cout.setf(std::ios::scientific);
    while (true)
    {
        for (int i = 0; i < bins; ++i)
            if (!(ff >> mag[static_cast<size_t>(i)]))
                return 0;
        for (int i = 0; i < bins; ++i)
            ff >> cosPhase[static_cast<size_t>(i)];
        for (int i = 0; i < bins; ++i)
            ff >> sinPhase[static_cast<size_t>(i)];
        if (!ff)
            return 6;

        model.processFrame(mag.data(), cosPhase.data(), sinPhase.data(), params, mask.data(), dphi.data());
        for (float v : mask)
            std::cout << v << '\n';
        for (float v : dphi)
            std::cout << v << '\n';
    }
}
}

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::cerr << "usage: mamba2_frame_validate model.json model.bin variant frames.txt\n";
        return 2;
    }

    std::ifstream jf(argv[1], std::ios::binary);
    std::ifstream bf(argv[2], std::ios::binary);
    if (!jf || !bf)
        return 3;

    std::vector<char> json((std::istreambuf_iterator<char>(jf)), {});
    std::vector<char> bin((std::istreambuf_iterator<char>(bf)), {});
    const std::string variant = argv[3];

    if (variant == "base")
        return runModel<Mamba2SpectralRawModel>(json, bin, argv[4]);
    if (variant == "xs")
        return runModel<Mamba2SizedSpectralRawModel<128, 256, 2, 512, 32>>(json, bin, argv[4]);
    if (variant == "xxs")
        return runModel<Mamba2SizedSpectralRawModel<64, 64, 1, 128, 16>>(json, bin, argv[4]);

    std::cerr << "unknown variant: " << variant << '\n';
    return 7;
}

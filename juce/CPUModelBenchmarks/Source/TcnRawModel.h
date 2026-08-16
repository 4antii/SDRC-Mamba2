#pragma once

#include "RawModelUtils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

class TcnRawModel
{
public:
    static constexpr int kParamDim = 4;
    static constexpr int kChannels = 32;
    static constexpr int kBlocks = 4;

    explicit TcnRawModel(int kernelSize = 5) : kernelSize_(kernelSize) {}

    void load(const TensorArchive& archive)
    {
        kernelSize_ = archive.getConfigInt("kernel_size", kernelSize_);
        dilationGrowth_ = archive.getConfigInt("dilation_growth", 10);
        gen0_.load(archive, "gen.0.weight", "gen.0.bias");
        gen2_.load(archive, "gen.2.weight", "gen.2.bias");
        gen4_.load(archive, "gen.4.weight", "gen.4.bias");

        for (int b = 0; b < kBlocks; ++b)
            blocks_[b].load(archive, "blocks." + std::to_string(b) + ".", b, kernelSize_, dilationForBlock(b));

        outputWeight_ = archive.get("output.weight").data;
        outputBias_ = archive.get("output.bias").data;
        contextSamples_ = receptiveField() - 1;
        reset();
    }

    void reset()
    {
        for (auto& block : blocks_)
            block.reset();
    }

    int latencySamples() const { return 0; }
    int receptiveContextSamples() const { return contextSamples_; }

    void processBlock(const float* input, float* output, int numSamples, const float* params)
    {
        makeCond(params);
        for (int n = 0; n < numSamples; ++n)
            output[n] = processSample(input[n]);
    }

private:
    struct Block
    {
        const float* convWeight = nullptr;
        const float* resWeight = nullptr;
        const float* reluWeight = nullptr;
        rawbench::BatchNorm1dEval bn;
        rawbench::Dense adaptor;
        int inChannels = 1;
        int kernel = 5;
        int dilation = 1;
        int ringSize = 1;
        int write = 0;
        std::vector<float> ring;
        std::array<float, kChannels * 2> gammaBeta {};

        void load(const TensorArchive& archive, const std::string& prefix, int index, int kernelSize, int dilationValue)
        {
            inChannels = index == 0 ? 1 : kChannels;
            kernel = kernelSize;
            dilation = dilationValue;
            ringSize = (kernel - 1) * dilation + 1;
            convWeight = archive.get(prefix + "conv1.weight").data;
            resWeight = archive.get(prefix + "res.weight").data;
            reluWeight = archive.get(prefix + "relu.weight").data;
            bn.load(archive, prefix + "film.bn.running_mean", prefix + "film.bn.running_var");
            adaptor.load(archive, prefix + "film.adaptor.weight", prefix + "film.adaptor.bias");
            ring.assign(static_cast<size_t>(inChannels * ringSize), 0.0f);
            write = 0;
        }

        void reset()
        {
            std::fill(ring.begin(), ring.end(), 0.0f);
            write = 0;
        }

        void updateCondition(const float* cond)
        {
            adaptor.forward(cond, gammaBeta.data());
        }

        void step(const float* input, float* output)
        {
            for (int ic = 0; ic < inChannels; ++ic)
                ring[static_cast<size_t>(ic * ringSize + write)] = input[ic];

            for (int oc = 0; oc < kChannels; ++oc)
            {
                float acc = 0.0f;
                for (int ic = 0; ic < inChannels; ++ic)
                {
                    const float* w = convWeight + (static_cast<size_t>(oc) * inChannels + ic) * kernel;
                    for (int k = 0; k < kernel; ++k)
                    {
                        const int delay = (kernel - 1 - k) * dilation;
                        int read = write - delay;
                        while (read < 0)
                            read += ringSize;
                        acc += w[k] * ring[static_cast<size_t>(ic * ringSize + read)];
                    }
                }

                acc = bn.apply(oc, acc);
                acc = acc * gammaBeta[oc] + gammaBeta[kChannels + oc];
                acc = acc >= 0.0f ? acc : reluWeight[oc] * acc;

                const int srcChannel = inChannels == 1 ? 0 : oc;
                int residualRead = write - 1;
                if (residualRead < 0)
                    residualRead += ringSize;
                const float residualInput = ring[static_cast<size_t>(srcChannel * ringSize + residualRead)];
                const float residual = resWeight[oc] * residualInput;
                output[oc] = acc + residual;
            }

            if (++write >= ringSize)
                write = 0;
        }
    };

    float processSample(float x)
    {
        blockIn1_[0] = x;
        blocks_[0].step(blockIn1_.data(), blockBufA_.data());
        blocks_[1].step(blockBufA_.data(), blockBufB_.data());
        blocks_[2].step(blockBufB_.data(), blockBufA_.data());
        blocks_[3].step(blockBufA_.data(), blockBufB_.data());

        float y = outputBias_[0];
        for (int c = 0; c < kChannels; ++c)
            y += outputWeight_[c] * blockBufB_[c];
        return std::tanh(y);
    }

    void makeCond(const float* params)
    {
        float h16[16];
        gen0_.forward(params, h16);
        for (float& v : h16)
            v = std::max(0.0f, v);
        float h32[32];
        gen2_.forward(h16, h32);
        for (float& v : h32)
            v = std::max(0.0f, v);
        gen4_.forward(h32, cond_.data());
        for (float& v : cond_)
            v = std::max(0.0f, v);

        for (auto& block : blocks_)
            block.updateCondition(cond_.data());
    }

    int dilationForBlock(int block) const
    {
        int d = 1;
        for (int i = 0; i < block; ++i)
            d *= dilationGrowth_;
        return d;
    }

    int receptiveField() const
    {
        int rf = kernelSize_;
        for (int b = 1; b < kBlocks; ++b)
            rf += (kernelSize_ - 1) * dilationForBlock(b);
        return rf;
    }

    int kernelSize_ = 5;
    int dilationGrowth_ = 10;
    int contextSamples_ = 0;
    rawbench::Dense gen0_, gen2_, gen4_;
    Block blocks_[kBlocks];
    const float* outputWeight_ = nullptr;
    const float* outputBias_ = nullptr;
    std::array<float, 32> cond_ {};
    std::array<float, 1> blockIn1_ {};
    std::array<float, kChannels> blockBufA_ {};
    std::array<float, kChannels> blockBufB_ {};
};

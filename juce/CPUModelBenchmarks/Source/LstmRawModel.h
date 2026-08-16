#pragma once

#include "RawModelUtils.h"

#include <algorithm>
#include <array>
#include <cmath>

class LstmRawModel
{
public:
    static constexpr int kParamDim = 4;
    static constexpr int kHidden = 32;

    void load(const TensorArchive& archive)
    {
        wIh_ = archive.get("lstm.weight_ih_l0").data;
        wHh_ = archive.get("lstm.weight_hh_l0").data;
        bIh_ = archive.get("lstm.bias_ih_l0").data;
        bHh_ = archive.get("lstm.bias_hh_l0").data;
        linear_.load(archive, "linear.weight", "linear.bias");
        reset();
    }

    void reset()
    {
        h_.fill(0.0f);
        c_.fill(0.0f);
    }

    void processBlock(const float* input, float* output, int numSamples, const float* params)
    {
        for (int n = 0; n < numSamples; ++n)
            output[n] = processSample(input[n], params);
    }

private:
    float processSample(float x, const float* params)
    {
        float in[5] = { x, params[0], params[1], params[2], params[3] };
        std::array<float, 4 * kHidden> gates {};

        for (int g = 0; g < 4 * kHidden; ++g)
        {
            float acc = bIh_[g] + bHh_[g];
            const float* rowIh = wIh_ + static_cast<size_t>(g) * 5u;
            const float* rowHh = wHh_ + static_cast<size_t>(g) * kHidden;
            for (int i = 0; i < 5; ++i)
                acc += rowIh[i] * in[i];
            for (int i = 0; i < kHidden; ++i)
                acc += rowHh[i] * h_[i];
            gates[g] = acc;
        }

        for (int i = 0; i < kHidden; ++i)
        {
            const float ingate = rawbench::sigmoid(gates[i]);
            const float forgetgate = rawbench::sigmoid(gates[kHidden + i]);
            const float cellgate = std::tanh(gates[2 * kHidden + i]);
            const float outgate = rawbench::sigmoid(gates[3 * kHidden + i]);
            c_[i] = forgetgate * c_[i] + ingate * cellgate;
            h_[i] = outgate * std::tanh(c_[i]);
        }

        float y = 0.0f;
        linear_.forward(h_.data(), &y);
        return std::tanh(y);
    }

    const float* wIh_ = nullptr;
    const float* wHh_ = nullptr;
    const float* bIh_ = nullptr;
    const float* bHh_ = nullptr;
    rawbench::Dense linear_;
    std::array<float, kHidden> h_ {};
    std::array<float, kHidden> c_ {};
};

#pragma once

#include "TensorArchive.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace rawbench
{
inline float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

inline float silu(float x)
{
    return x * sigmoid(x);
}

inline float softplus(float x)
{
    if (x > 20.0f)
        return x;
    return std::log1p(std::exp(x));
}

inline float gelu(float x)
{
    constexpr float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

inline float softsign(float x)
{
    return x / (1.0f + std::abs(x));
}

inline void prelu(float* x, int n, const float* alpha)
{
    for (int i = 0; i < n; ++i)
        x[i] = x[i] >= 0.0f ? x[i] : alpha[i] * x[i];
}

inline void preluShared(float* x, int n, float alpha)
{
    for (int i = 0; i < n; ++i)
        x[i] = x[i] >= 0.0f ? x[i] : alpha * x[i];
}

struct Dense
{
    const float* weight = nullptr;
    const float* bias = nullptr;
    int outDim = 0;
    int inDim = 0;

    void load(const TensorArchive& archive, const std::string& weightName, const std::string& biasName = {})
    {
        const auto& w = archive.get(weightName);
        weight = w.data;
        outDim = w.shape.at(0);
        inDim = w.shape.at(1);
        if (!biasName.empty() && archive.contains(biasName))
            bias = archive.get(biasName).data;
        else
            bias = nullptr;
    }

    void forward(const float* input, float* output) const
    {
        for (int o = 0; o < outDim; ++o)
        {
            float acc = bias != nullptr ? bias[o] : 0.0f;
            const float* row = weight + static_cast<size_t>(o) * static_cast<size_t>(inDim);
            for (int i = 0; i < inDim; ++i)
                acc += row[i] * input[i];
            output[o] = acc;
        }
    }
};

struct LayerNorm
{
    const float* weight = nullptr;
    const float* bias = nullptr;
    int dim = 0;

    void load(const TensorArchive& archive, const std::string& weightName, const std::string& biasName)
    {
        const auto& w = archive.get(weightName);
        weight = w.data;
        bias = archive.get(biasName).data;
        dim = w.count;
    }

    void apply(const float* input, float* output) const
    {
        float mean = 0.0f;
        for (int i = 0; i < dim; ++i)
            mean += input[i];
        mean /= static_cast<float>(dim);

        float var = 0.0f;
        for (int i = 0; i < dim; ++i)
        {
            const float d = input[i] - mean;
            var += d * d;
        }
        var /= static_cast<float>(dim);
        const float inv = 1.0f / std::sqrt(var + 1.0e-5f);

        for (int i = 0; i < dim; ++i)
            output[i] = (input[i] - mean) * inv * weight[i] + bias[i];
    }
};

struct BatchNorm1dEval
{
    const float* mean = nullptr;
    const float* var = nullptr;
    int dim = 0;

    void load(const TensorArchive& archive, const std::string& meanName, const std::string& varName)
    {
        const auto& m = archive.get(meanName);
        mean = m.data;
        var = archive.get(varName).data;
        dim = m.count;
    }

    float apply(int channel, float x) const
    {
        return (x - mean[channel]) / std::sqrt(var[channel] + 1.0e-5f);
    }
};
}

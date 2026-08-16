#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <string>
#include <vector>

#include "TensorArchive.h"

class S4RawModel
{
public:
    static constexpr int kParams = 4;
    static constexpr int kHidden = 32;
    static constexpr int kState = 2;
    static constexpr int kBlocks = 4;

    void load(const TensorArchive& archive)
    {
        mlp0.load(archive, "control_parameter_mlp.0.weight", "control_parameter_mlp.0.bias");
        mlp1.load(archive, "control_parameter_mlp.2.weight", "control_parameter_mlp.2.bias");
        mlp2.load(archive, "control_parameter_mlp.4.weight", "control_parameter_mlp.4.bias");
        expand.load(archive, "expand.weight", "expand.bias");
        contract.load(archive, "contract.weight", "contract.bias");
        for (int i = 0; i < kBlocks; ++i)
            blocks[static_cast<size_t>(i)].load(archive, "blocks." + std::to_string(i) + ".");
        reset();
    }

    void reset()
    {
        for (auto& block : blocks)
            block.reset();
    }

    void processBlock(const float* input, float* output, int numSamples, const float* paramsNorm)
    {
        computeCondition(paramsNorm);

        for (int t = 0; t < numSamples; ++t)
        {
            std::array<float, kHidden> x {};
            std::array<float, kHidden> y {};

            for (int h = 0; h < kHidden; ++h)
                x[h] = input[t] * expand.weight[h] + expand.bias[h];

            for (auto& block : blocks)
            {
                block.step(x.data(), y.data(), cond.data());
                x = y;
            }

            float out = contract.bias[0];
            for (int h = 0; h < kHidden; ++h)
                out += contract.weight[h] * x[h];
            output[t] = std::tanh(out);
        }
    }

private:
    struct Dense
    {
        const float* weight = nullptr;
        const float* bias = nullptr;
        int outDim = 0;
        int inDim = 0;

        void load(const TensorArchive& archive, const std::string& weightName, const std::string& biasName)
        {
            const auto& w = archive.get(weightName);
            weight = w.data;
            outDim = w.shape[0];
            inDim = w.shape[1];
            bias = archive.get(biasName).data;
        }

        void forward(const float* input, float* output) const
        {
            for (int o = 0; o < outDim; ++o)
            {
                float y = bias[o];
                const float* row = weight + static_cast<size_t>(o) * static_cast<size_t>(inDim);
                for (int i = 0; i < inDim; ++i)
                    y += row[i] * input[i];
                output[o] = y;
            }
        }
    };

    struct S4Block
    {
        Dense linear;
        Dense filmAdaptor;
        const float* activation1 = nullptr;
        const float* activation2 = nullptr;
        const float* dSkip = nullptr;
        const float* residualWeight = nullptr;
        const float* bnMean = nullptr;
        const float* bnVar = nullptr;

        std::array<std::complex<float>, kHidden * kState> aBar {};
        std::array<std::complex<float>, kHidden * kState> bBar {};
        std::array<std::complex<float>, kHidden * kState> cBar {};
        std::array<std::complex<float>, kHidden * kState> state {};
        std::array<float, kHidden * 2> gammaBeta {};
        std::array<float, kHidden> linearOut {};
        std::array<float, kHidden> s4Out {};

        void load(const TensorArchive& archive, const std::string& prefix)
        {
            linear.load(archive, prefix + "linear.weight", prefix + "linear.bias");
            activation1 = archive.get(prefix + "activation1.weight").data;
            activation2 = archive.get(prefix + "activation2.weight").data;
            dSkip = archive.get(prefix + "s4.D").data;
            const float* invDt = archive.get(prefix + "s4.kernel.inv_dt").data;
            const float* cTensor = archive.get(prefix + "s4.kernel.C").data;
            const float* bTensor = archive.get(prefix + "s4.kernel.B").data;
            const float* aReal = archive.get(prefix + "s4.kernel.A_real").data;
            const float* aImag = archive.get(prefix + "s4.kernel.A_imag").data;
            bnMean = archive.get(prefix + "batchnorm.running_mean").data;
            bnVar = archive.get(prefix + "batchnorm.running_var").data;
            filmAdaptor.load(archive, prefix + "film.conditional_information_adaptor.weight",
                             prefix + "film.conditional_information_adaptor.bias");
            residualWeight = archive.get(prefix + "residual_connection.weight").data;

            for (int h = 0; h < kHidden; ++h)
            {
                const float dt = std::exp(invDt[h]);
                for (int n = 0; n < kState; ++n)
                {
                    const auto A = std::complex<float>(-std::exp(aReal[h * kState + n]),
                                                       -aImag[h * kState + n]);
                    const auto B = getComplex(bTensor, h, n);
                    const auto C = getComplex(cTensor, h, n);
                    const auto dtA = dt * A;
                    const auto e = std::exp(dtA);
                    const size_t i = index(h, n);
                    aBar[i] = e;
                    bBar[i] = B * ((e - std::complex<float>(1.0f, 0.0f)) / A);
                    cBar[i] = C;
                }
            }
            reset();
        }

        void reset()
        {
            state.fill({ 0.0f, 0.0f });
        }

        void step(const float* input, float* output, const float* cond)
        {
            filmAdaptor.forward(cond, gammaBeta.data());

            for (int h = 0; h < kHidden; ++h)
            {
                float y = linear.bias[h];
                const float* row = linear.weight + static_cast<size_t>(h) * kHidden;
                for (int i = 0; i < kHidden; ++i)
                    y += row[i] * input[i];
                linearOut[h] = prelu(y, activation1[0]);
            }

            for (int h = 0; h < kHidden; ++h)
            {
                std::complex<float> acc { 0.0f, 0.0f };
                const float u = linearOut[h];
                for (int n = 0; n < kState; ++n)
                {
                    const size_t i = index(h, n);
                    state[i] = aBar[i] * state[i] + bBar[i] * u;
                    acc += cBar[i] * state[i];
                }
                s4Out[h] = (2.0f * acc.real()) + dSkip[h] * u;
            }

            for (int h = 0; h < kHidden; ++h)
            {
                const float invStd = 1.0f / std::sqrt(bnVar[h] + 1.0e-5f);
                const float g = gammaBeta[h];
                const float b = gammaBeta[kHidden + h];
                float y = (s4Out[h] - bnMean[h]) * invStd;
                y = y * g + b;
                y = prelu(y, activation2[0]);
                output[h] = y + residualWeight[h] * input[h];
            }
        }

        static std::complex<float> getComplex(const float* tensor, int h, int n)
        {
            const size_t base = ((static_cast<size_t>(h) * kState) + static_cast<size_t>(n)) * 2u;
            return { tensor[base], tensor[base + 1] };
        }

        static size_t index(int h, int n)
        {
            return static_cast<size_t>(h) * kState + static_cast<size_t>(n);
        }
    };

    void computeCondition(const float* paramsNorm)
    {
        mlp0.forward(paramsNorm, mlpHidden0.data());
        for (float& v : mlpHidden0)
            v = std::max(0.0f, v);
        mlp1.forward(mlpHidden0.data(), mlpHidden1.data());
        for (float& v : mlpHidden1)
            v = std::max(0.0f, v);
        mlp2.forward(mlpHidden1.data(), cond.data());
        for (float& v : cond)
            v = std::max(0.0f, v);
    }

    static float prelu(float x, float alpha)
    {
        return x >= 0.0f ? x : alpha * x;
    }

    Dense mlp0, mlp1, mlp2, expand, contract;
    S4Block blocks[kBlocks];
    std::array<float, 16> mlpHidden0 {};
    std::array<float, 32> mlpHidden1 {};
    std::array<float, 32> cond {};
};

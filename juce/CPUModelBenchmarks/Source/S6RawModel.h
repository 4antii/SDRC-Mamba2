#pragma once

#include "RawModelUtils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>

class S6RawModel
{
public:
    static constexpr int kParamDim = 4;
    static constexpr int kWindow = 64;
    static constexpr int kNfft = 128;
    static constexpr int kDModel = 2;
    static constexpr int kInner = 4;
    static constexpr int kState = 16;
    static constexpr int kDConv = 4;

    void load(const TensorArchive& archive)
    {
        fcIn_.load(archive, "fc_in.weight", "fc_in.bias");
        mamba1_.load(archive, "mamba1.core.");
        fcAfterM1_.load(archive, "fc_after_m1.0.weight", "fc_after_m1.0.bias");
        freqConvWeight_ = archive.get("freq_conv.weight").data;
        freqConvBias_ = archive.get("freq_conv.bias").data;
        filmAffine_.load(archive, "film.affine.weight", "film.affine.bias");
        filmPost_.load(archive, "film.post.weight", "film.post.bias");
        tfilmGru_.load(archive, "tfilm.gru.");
        tfilmAffine_.load(archive, "tfilm.affine.weight", "tfilm.affine.bias");
        tfilmPost_.load(archive, "tfilm.post.weight", "tfilm.post.bias");
        mamba2_.load(archive, "mamba2.core.");
        fcAfterM2_.load(archive, "fc_after_m2.0.weight", "fc_after_m2.0.bias");
        outHead_.load(archive, "out_head.weight", "out_head.bias");
        reset();
    }

    void reset()
    {
        window_.fill(0.0f);
        orderedWindow_.fill(0.0f);
        dft_.fill({ 0.0f, 0.0f });
        windowWrite_ = 0;
        mamba1_.reset();
        mamba2_.reset();
        tfilmGru_.reset();
    }

    void processBlock(const float* input, float* output, int numSamples, const float* params)
    {
        for (int n = 0; n < numSamples; ++n)
            output[n] = processSample(input[n], params);
    }

private:
    struct MambaCore
    {
        rawbench::Dense inProj;
        rawbench::Dense xProj;
        rawbench::Dense dtProj;
        rawbench::Dense outProj;
        const float* convWeight = nullptr;
        const float* convBias = nullptr;
        const float* aLog = nullptr;
        std::array<float, kInner * kState> aNeg {};
        const float* dSkip = nullptr;
        std::array<float, kInner * kDConv> convState {};
        std::array<float, kInner * kState> ssmState {};

        void load(const TensorArchive& archive, const std::string& prefix)
        {
            inProj.load(archive, prefix + "in_proj.weight");
            convWeight = archive.get(prefix + "conv1d.weight").data;
            convBias = archive.get(prefix + "conv1d.bias").data;
            xProj.load(archive, prefix + "x_proj.weight");
            dtProj.load(archive, prefix + "dt_proj.weight", prefix + "dt_proj.bias");
            aLog = archive.get(prefix + "A_log").data;
            dSkip = archive.get(prefix + "D").data;
            outProj.load(archive, prefix + "out_proj.weight");
            for (int i = 0; i < kInner * kState; ++i)
                aNeg[static_cast<size_t>(i)] = -std::exp(aLog[i]);
        }

        void reset()
        {
            convState.fill(0.0f);
            ssmState.fill(0.0f);
        }

        void step(const float* input, float* output)
        {
            float xz[2 * kInner];
            inProj.forward(input, xz);
            const float* xIn = xz;
            const float* z = xz + kInner;

            float x[kInner];
            for (int d = 0; d < kInner; ++d)
            {
                float* state = convState.data() + d * kDConv;
                for (int i = 0; i < kDConv - 1; ++i)
                    state[i] = state[i + 1];
                state[kDConv - 1] = xIn[d];

                const float* w = convWeight + d * kDConv;
                float acc = convBias[d];
                for (int i = 0; i < kDConv; ++i)
                    acc += state[i] * w[i];
                x[d] = rawbench::silu(acc);
            }

            float xDb[1 + 2 * kState];
            xProj.forward(x, xDb);
            const float dtIn = xDb[0];
            const float* B = xDb + 1;
            const float* C = xDb + 1 + kState;

            float scan[kInner];
            for (int d = 0; d < kInner; ++d)
            {
                const float dt = rawbench::softplus(dtProj.weight[d] * dtIn + dtProj.bias[d]);
                float acc = 0.0f;
                float* st = ssmState.data() + d * kState;
                for (int s = 0; s < kState; ++s)
                {
                    const float A = aNeg[static_cast<size_t>(d * kState + s)];
                    st[s] = st[s] * std::exp(dt * A) + x[d] * (dt * B[s]);
                    acc += st[s] * C[s];
                }
                scan[d] = (acc + dSkip[d] * x[d]) * rawbench::silu(z[d]);
            }

            outProj.forward(scan, output);
        }
    };

    struct Gru2
    {
        const float* wIh = nullptr;
        const float* wHh = nullptr;
        const float* bIh = nullptr;
        const float* bHh = nullptr;
        std::array<float, kDModel> h {};

        void load(const TensorArchive& archive, const std::string& prefix)
        {
            wIh = archive.get(prefix + "weight_ih_l0").data;
            wHh = archive.get(prefix + "weight_hh_l0").data;
            bIh = archive.get(prefix + "bias_ih_l0").data;
            bHh = archive.get(prefix + "bias_hh_l0").data;
        }

        void reset() { h.fill(0.0f); }

        void step(const float* input6, float* output2)
        {
            float gi[6];
            float gh[6];
            for (int g = 0; g < 6; ++g)
            {
                float ai = bIh[g];
                const float* rowI = wIh + g * 6;
                for (int i = 0; i < 6; ++i)
                    ai += rowI[i] * input6[i];
                gi[g] = ai;

                float ah = bHh[g];
                const float* rowH = wHh + g * kDModel;
                for (int i = 0; i < kDModel; ++i)
                    ah += rowH[i] * h[i];
                gh[g] = ah;
            }

            for (int i = 0; i < kDModel; ++i)
            {
                const float r = rawbench::sigmoid(gi[i] + gh[i]);
                const float z = rawbench::sigmoid(gi[kDModel + i] + gh[kDModel + i]);
                const float n = std::tanh(gi[2 * kDModel + i] + r * gh[2 * kDModel + i]);
                output2[i] = (1.0f - z) * n + z * h[i];
            }
            h[0] = output2[0];
            h[1] = output2[1];
        }
    };

    float processSample(float sample, const float* params)
    {
        const float oldSample = window_[static_cast<size_t>(windowWrite_)];
        window_[static_cast<size_t>(windowWrite_)] = sample;
        if (++windowWrite_ >= kWindow)
            windowWrite_ = 0;
        updateDft(oldSample, sample);
        makeOrderedWindow();

        float h[kDModel];
        float tmp[kDModel];
        fcIn_.forward(orderedWindow_.data(), h);
        mamba1_.step(h, h);
        fcAfterM1_.forward(h, tmp);
        for (int i = 0; i < kDModel; ++i)
            h[i] = rawbench::gelu(tmp[i]);

        float features[2];
        fftFeatures(features);
        float cond[6] = { params[0], params[1], params[2], params[3], features[0], features[1] };

        float gb[4];
        filmAffine_.forward(cond, gb);
        float filmIn[kDModel] = { gb[0] * h[0] + gb[2], gb[1] * h[1] + gb[3] };
        filmPost_.forward(filmIn, gb);
        h[0] = gb[0] * rawbench::softsign(gb[2]);
        h[1] = gb[1] * rawbench::softsign(gb[3]);

        float gruH[kDModel];
        tfilmGru_.step(cond, gruH);
        tfilmAffine_.forward(gruH, gb);
        float tfIn[kDModel] = { gb[0] * h[0] + gb[2], gb[1] * h[1] + gb[3] };
        tfilmPost_.forward(tfIn, gb);
        h[0] = gb[0] * rawbench::softsign(gb[2]);
        h[1] = gb[1] * rawbench::softsign(gb[3]);

        mamba2_.step(h, h);
        fcAfterM2_.forward(h, tmp);
        for (int i = 0; i < kDModel; ++i)
            h[i] = rawbench::gelu(tmp[i]);

        float gain = 0.0f;
        outHead_.forward(h, &gain);
        return gain * sample;
    }

    void fftFeatures(float* out)
    {
        for (int oc = 0; oc < 2; ++oc)
        {
            float acc = freqConvBias_[oc];
            const float* w = freqConvWeight_ + oc * kNfft;
            for (int k = 0; k < kNfft; ++k)
            {
                const auto z = dft_[static_cast<size_t>(k)];
                acc += w[k] * std::sqrt(z.real() * z.real() + z.imag() * z.imag());
            }
            out[oc] = acc;
        }
    }

    void updateDft(float oldSample, float newSample)
    {
        ensureDftTables();
        for (int k = 0; k < kNfft; ++k)
        {
            const size_t i = static_cast<size_t>(k);
            dft_[i] = rot_[i] * (dft_[i] - std::complex<float>(oldSample, 0.0f))
                    + tail_[i] * newSample;
        }
    }

    void makeOrderedWindow()
    {
        int idx = windowWrite_;
        for (int i = 0; i < kWindow; ++i)
        {
            orderedWindow_[static_cast<size_t>(i)] = window_[static_cast<size_t>(idx)];
            if (++idx >= kWindow)
                idx = 0;
        }
    }

    static void ensureDftTables()
    {
        if (dftTablesReady_)
            return;

        constexpr float twoPi = 6.28318530717958647692f;
        for (int k = 0; k < kNfft; ++k)
        {
            const float omega = twoPi * static_cast<float>(k) / static_cast<float>(kNfft);
            rot_[static_cast<size_t>(k)] = { std::cos(omega), std::sin(omega) };
            const float tailPhase = -omega * static_cast<float>(kWindow - 1);
            tail_[static_cast<size_t>(k)] = { std::cos(tailPhase), std::sin(tailPhase) };
        }
        dftTablesReady_ = true;
    }

    rawbench::Dense fcIn_;
    MambaCore mamba1_;
    rawbench::Dense fcAfterM1_;
    const float* freqConvWeight_ = nullptr;
    const float* freqConvBias_ = nullptr;
    rawbench::Dense filmAffine_;
    rawbench::Dense filmPost_;
    Gru2 tfilmGru_;
    rawbench::Dense tfilmAffine_;
    rawbench::Dense tfilmPost_;
    MambaCore mamba2_;
    rawbench::Dense fcAfterM2_;
    rawbench::Dense outHead_;
    std::array<float, kWindow> window_ {};
    std::array<float, kWindow> orderedWindow_ {};
    std::array<std::complex<float>, kNfft> dft_ {};
    int windowWrite_ = 0;

    static inline bool dftTablesReady_ = false;
    static inline std::array<std::complex<float>, kNfft> rot_ {};
    static inline std::array<std::complex<float>, kNfft> tail_ {};
};

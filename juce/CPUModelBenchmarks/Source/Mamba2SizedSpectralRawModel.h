#pragma once

#include "RawModelUtils.h"

#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

template <int DModel, int FilmHidden, int Depth, int HeadHidden, int HeadDim>
class Mamba2SizedSpectralRawModel
{
public:
    static constexpr int kFreqBins = 257;
    static constexpr int kPhaseFeat = 514;
    static constexpr int kParamDim = 4;
    static constexpr int kDModel = DModel;
    static constexpr int kFilmHidden = FilmHidden;
    static constexpr int kDepth = Depth;
    static constexpr int kHeadHidden = HeadHidden;
    static constexpr int kInner = 2 * DModel;
    static constexpr int kStateDim = 128;
    static constexpr int kConvDim = kInner + 2 * kStateDim;
    static constexpr int kHeadDim = HeadDim;
    static constexpr int kNHeads = kInner / kHeadDim;
    static constexpr int kZxbcdt = kInner + kConvDim + kNHeads;

    void load(const TensorArchive& archive)
    {
        inProjMag_.load(archive, "in_proj_mag.weight", "in_proj_mag.bias");
        inProjPhase_.load(archive, "in_proj_phase.weight", "in_proj_phase.bias");
        for (int i = 0; i < kDepth; ++i)
        {
            blocksMag_[static_cast<size_t>(i)].load(archive, "blocks_mag." + std::to_string(i) + ".");
            blocksPh_[static_cast<size_t>(i)].load(archive, "blocks_ph." + std::to_string(i) + ".");
        }
        postNormMag_.load(archive, "post_norm_mag.weight", "post_norm_mag.bias");
        postNormPh_.load(archive, "post_norm_ph.weight", "post_norm_ph.bias");
        stackGateMag_ = archive.get("stack_gate_mag").data[0];
        stackGatePh_ = archive.get("stack_gate_ph").data[0];
        filmBeforeHeadMag_.load(archive, "film_before_head_mag.");
        filmBeforeHeadPh_.load(archive, "film_before_head_ph.");
        headMag0_.load(archive, "head_mag.0.weight", "head_mag.0.bias");
        headMagAlpha_ = archive.get("head_mag.1.weight").data[0];
        headMag2_.load(archive, "head_mag.2.weight", "head_mag.2.bias");
        outScaleMag_ = archive.get("out_scale_mag").data[0];
        headPh0_.load(archive, "head_ph.0.weight", "head_ph.0.bias");
        headPhAlpha_ = archive.get("head_ph.1.weight").data[0];
        headPh2_.load(archive, "head_ph.2.weight", "head_ph.2.bias");
        reset();
    }

    void reset()
    {
        for (auto& b : blocksMag_)
            b.reset();
        for (auto& b : blocksPh_)
            b.reset();
    }

    void processFrame(const float* magLin, const float* cosPhase, const float* sinPhase,
                      const float* params, float* outMask, float* outDphi)
    {
        for (int i = 0; i < kFreqBins; ++i)
            magFeat_[static_cast<size_t>(i)] = std::log1p(magLin[i]);

        inProjMag_.forward(magFeat_.data(), hInMag_.data());
        std::memcpy(hMag_.data(), hInMag_.data(), sizeof(float) * kDModel);
        for (auto& block : blocksMag_)
            block.forward(hMag_.data(), params, hMag_.data());
        postNormMag_.apply(hMag_.data(), postBuf_.data());
        for (int i = 0; i < kDModel; ++i)
            hMag_[static_cast<size_t>(i)] = hInMag_[static_cast<size_t>(i)] + stackGateMag_ * postBuf_[static_cast<size_t>(i)];
        filmBeforeHeadMag_.apply(hMag_.data(), params, headIn_.data());
        headMag0_.forward(headIn_.data(), headHidden_.data());
        rawbench::preluShared(headHidden_.data(), kHeadHidden, headMagAlpha_);
        headMag2_.forward(headHidden_.data(), logits_.data());
        for (int i = 0; i < kFreqBins; ++i)
            outMask[i] = rawbench::sigmoid(logits_[static_cast<size_t>(i)]) * outScaleMag_;

        for (int i = 0; i < kFreqBins; ++i)
        {
            phaseFeat_[static_cast<size_t>(i)] = cosPhase[i];
            phaseFeat_[static_cast<size_t>(kFreqBins + i)] = sinPhase[i];
        }
        inProjPhase_.forward(phaseFeat_.data(), hInPh_.data());
        std::memcpy(hPh_.data(), hInPh_.data(), sizeof(float) * kDModel);
        for (auto& block : blocksPh_)
            block.forward(hPh_.data(), params, hPh_.data());
        postNormPh_.apply(hPh_.data(), postBuf_.data());
        for (int i = 0; i < kDModel; ++i)
            hPh_[static_cast<size_t>(i)] = hInPh_[static_cast<size_t>(i)] + stackGatePh_ * postBuf_[static_cast<size_t>(i)];
        filmBeforeHeadPh_.apply(hPh_.data(), params, headIn_.data());
        headPh0_.forward(headIn_.data(), headHidden_.data());
        rawbench::preluShared(headHidden_.data(), kHeadHidden, headPhAlpha_);
        headPh2_.forward(headHidden_.data(), logits_.data());
        constexpr float pi = 3.14159265358979323846f;
        for (int i = 0; i < kFreqBins; ++i)
            outDphi[i] = pi * std::tanh(logits_[static_cast<size_t>(i)]);
    }

private:
    struct RMSNormGated
    {
        const float* weight = nullptr;
        int dim = 0;

        void load(const TensorArchive& archive, const std::string& weightName)
        {
            const auto& w = archive.get(weightName);
            weight = w.data;
            dim = w.count;
        }

        void apply(const float* input, const float* gate, float* output) const
        {
            float variance = 0.0f;
            for (int i = 0; i < dim; ++i)
            {
                const float v = input[i] * rawbench::silu(gate[i]);
                output[i] = v;
                variance += v * v;
            }
            variance /= static_cast<float>(dim);
            const float scale = 1.0f / std::sqrt(variance + 1.0e-5f);
            for (int i = 0; i < dim; ++i)
                output[i] = output[i] * scale * weight[i];
        }
    };

    struct FiLM
    {
        rawbench::Dense fc0;
        rawbench::Dense fc1;
        std::array<float, kFilmHidden> hidden {};
        std::array<float, kDModel * 2> gb {};

        void load(const TensorArchive& archive, const std::string& prefix)
        {
            fc0.load(archive, prefix + "net.0.weight", prefix + "net.0.bias");
            fc1.load(archive, prefix + "net.2.weight", prefix + "net.2.bias");
        }

        void apply(const float* input, const float* params, float* output)
        {
            fc0.forward(params, hidden.data());
            for (float& v : hidden)
                v = rawbench::silu(v);
            fc1.forward(hidden.data(), gb.data());
            for (int i = 0; i < kDModel; ++i)
                output[i] = input[i] * (1.0f + gb[static_cast<size_t>(i)]) + gb[static_cast<size_t>(kDModel + i)];
        }
    };

    struct MambaCore
    {
        rawbench::Dense inProj;
        const float* convWeight = nullptr;
        const float* convBias = nullptr;
        const float* dtBias = nullptr;
        const float* aLog = nullptr;
        const float* dSkip = nullptr;
        RMSNormGated rmsNorm;
        rawbench::Dense outProj;
        std::vector<float> convState;
        std::vector<float> ssmState;
        std::array<float, kZxbcdt> zxbcdt {};
        std::array<float, kConvDim> convOut {};
        std::array<float, kInner> scanOut {};
        std::array<float, kInner> normOut {};
        std::array<float, kDModel> outBuf {};
        std::array<float, kStateDim> scaledB {};

        MambaCore()
            : convState(static_cast<size_t>(kConvDim * 4), 0.0f),
              ssmState(static_cast<size_t>(kNHeads * kHeadDim * kStateDim), 0.0f)
        {
        }

        void load(const TensorArchive& archive, const std::string& prefix)
        {
            inProj.load(archive, prefix + "in_proj.weight");
            convWeight = archive.get(prefix + "conv1d.weight").data;
            convBias = archive.get(prefix + "conv1d.bias").data;
            dtBias = archive.get(prefix + "dt_bias").data;
            aLog = archive.get(prefix + "A_log").data;
            dSkip = archive.get(prefix + "D").data;
            rmsNorm.load(archive, prefix + "norm.weight");
            outProj.load(archive, prefix + "out_proj.weight");
        }

        void reset()
        {
            std::fill(convState.begin(), convState.end(), 0.0f);
            std::fill(ssmState.begin(), ssmState.end(), 0.0f);
        }

        void step(const float* input, float* output)
        {
            inProj.forward(input, zxbcdt.data());
            const float* z = zxbcdt.data();
            const float* xBCIn = zxbcdt.data() + kInner;
            const float* dtIn = zxbcdt.data() + kInner + kConvDim;

            for (int i = 0; i < kConvDim; ++i)
            {
                float* state = convState.data() + static_cast<size_t>(i * 4);
                state[0] = state[1];
                state[1] = state[2];
                state[2] = state[3];
                state[3] = xBCIn[i];
                const float* w = convWeight + static_cast<size_t>(i * 4);
                float y = convBias[i];
                for (int k = 0; k < 4; ++k)
                    y += state[k] * w[k];
                convOut[static_cast<size_t>(i)] = rawbench::silu(y);
            }

            const float* x = convOut.data();
            const float* B = convOut.data() + kInner;
            const float* C = convOut.data() + kInner + kStateDim;

            for (int h = 0; h < kNHeads; ++h)
            {
                const float dt = rawbench::softplus(dtIn[h] + dtBias[h]);
                const float dA = std::exp(dt * (-std::exp(aLog[h])));
                const float dScale = dSkip[h];
                for (int s = 0; s < kStateDim; ++s)
                    scaledB[static_cast<size_t>(s)] = B[s] * dt;

                for (int p = 0; p < kHeadDim; ++p)
                {
                    const float xhp = x[h * kHeadDim + p];
                    float* state = ssmState.data() + static_cast<size_t>((h * kHeadDim + p) * kStateDim);
                    float acc = 0.0f;
                    for (int s = 0; s < kStateDim; ++s)
                    {
                        state[s] = state[s] * dA + scaledB[static_cast<size_t>(s)] * xhp;
                        acc += state[s] * C[s];
                    }
                    scanOut[static_cast<size_t>(h * kHeadDim + p)] = acc + dScale * xhp;
                }
            }

            rmsNorm.apply(scanOut.data(), z, normOut.data());
            outProj.forward(normOut.data(), outBuf.data());
            std::memcpy(output, outBuf.data(), sizeof(float) * kDModel);
        }
    };

    struct MambaBlock
    {
        rawbench::LayerNorm norm;
        FiLM film;
        MambaCore mamba;
        std::array<float, kDModel> normBuf {};
        std::array<float, kDModel> filmBuf {};
        std::array<float, kDModel> mambaBuf {};

        void load(const TensorArchive& archive, const std::string& prefix)
        {
            norm.load(archive, prefix + "norm.weight", prefix + "norm.bias");
            film.load(archive, prefix + "film.");
            mamba.load(archive, prefix + "mamba.");
        }

        void reset() { mamba.reset(); }

        void forward(const float* input, const float* params, float* output)
        {
            norm.apply(input, normBuf.data());
            film.apply(normBuf.data(), params, filmBuf.data());
            mamba.step(filmBuf.data(), mambaBuf.data());
            for (int i = 0; i < kDModel; ++i)
                output[i] = input[i] + mambaBuf[static_cast<size_t>(i)];
        }
    };

    rawbench::Dense inProjMag_;
    rawbench::Dense inProjPhase_;
    std::array<MambaBlock, kDepth> blocksMag_;
    std::array<MambaBlock, kDepth> blocksPh_;
    rawbench::LayerNorm postNormMag_;
    rawbench::LayerNorm postNormPh_;
    float stackGateMag_ = 1.0f;
    float stackGatePh_ = 1.0f;
    FiLM filmBeforeHeadMag_;
    FiLM filmBeforeHeadPh_;
    rawbench::Dense headMag0_;
    rawbench::Dense headMag2_;
    rawbench::Dense headPh0_;
    rawbench::Dense headPh2_;
    float headMagAlpha_ = 0.25f;
    float headPhAlpha_ = 0.25f;
    float outScaleMag_ = 2.0f;

    std::array<float, kFreqBins> magFeat_ {};
    std::array<float, kPhaseFeat> phaseFeat_ {};
    std::array<float, kDModel> hInMag_ {};
    std::array<float, kDModel> hMag_ {};
    std::array<float, kDModel> hInPh_ {};
    std::array<float, kDModel> hPh_ {};
    std::array<float, kDModel> postBuf_ {};
    std::array<float, kDModel> headIn_ {};
    std::array<float, kHeadHidden> headHidden_ {};
    std::array<float, kFreqBins> logits_ {};
};

#pragma once

#include <juce_core/juce_core.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

class TensorArchive
{
public:
    struct TensorView
    {
        const float* data = nullptr;
        std::vector<int> shape;
        int count = 0;
    };

    bool load(const void* jsonData, size_t jsonSize, const void* binData, size_t binSize)
    {
        try
        {
            juce::MemoryInputStream stream(jsonData, jsonSize, false);
            const auto jsonText = stream.readEntireStreamAsString().toStdString();
            auto result = juce::JSON::parse(jsonText, metadata_);
            if (result.failed())
                return false;

            auto* rootObj = metadata_.getDynamicObject();
            if (rootObj == nullptr)
                return false;

            const auto tensorsVar = rootObj->getProperty("tensors");
            auto* tensorsObj = tensorsVar.getDynamicObject();
            if (tensorsObj == nullptr)
                return false;

            if ((binSize % sizeof(float)) != 0)
                return false;

            const size_t floatCount = binSize / sizeof(float);
            blob_.resize(floatCount);
            std::memcpy(blob_.data(), binData, binSize);

            tensors_.clear();
            for (const auto& entry : tensorsObj->getProperties())
            {
                auto* metaObj = entry.value.getDynamicObject();
                if (metaObj == nullptr)
                    return false;

                const int offset = static_cast<int>(metaObj->getProperty("offset_floats"));
                const int count = static_cast<int>(metaObj->getProperty("num_floats"));
                if (offset < 0 || count < 0 || static_cast<size_t>(offset + count) > blob_.size())
                    return false;

                TensorView view;
                view.data = blob_.data() + offset;
                view.count = count;
                if (const auto* shapeArray = metaObj->getProperty("shape").getArray())
                {
                    view.shape.reserve(static_cast<size_t>(shapeArray->size()));
                    for (const auto& dim : *shapeArray)
                        view.shape.push_back(static_cast<int>(dim));
                }
                tensors_.emplace(entry.name.toString().toStdString(), std::move(view));
            }

            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    int getConfigInt(const juce::String& key, int defaultValue) const
    {
        if (auto* rootObj = metadata_.getDynamicObject())
            if (auto* configObj = rootObj->getProperty("config").getDynamicObject())
                if (configObj->hasProperty(key))
                    return static_cast<int>(configObj->getProperty(key));
        return defaultValue;
    }

    const TensorView& get(const std::string& name) const
    {
        return tensors_.at(name);
    }

    bool contains(const std::string& name) const
    {
        return tensors_.find(name) != tensors_.end();
    }

private:
    juce::var metadata_;
    std::vector<float> blob_;
    std::unordered_map<std::string, TensorView> tensors_;
};

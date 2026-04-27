/*
 * ScatterAdd TensorRT Plugin Implementation
 */

#include "scatter_add_plugin.h"
#include "scatter_add_kernel.h"
#include <cstring>
#include <cassert>
#include <iostream>

namespace planr1 {
namespace plugin {

// Static members initialization
nvinfer1::PluginFieldCollection ScatterAddPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> ScatterAddPluginCreator::mPluginAttributes;

// ============================================================================
// ScatterAddPlugin Implementation
// ============================================================================

ScatterAddPlugin::ScatterAddPlugin(int axis)
    : mAxis(axis)
    , mNamespace(SCATTER_ADD_PLUGIN_NAMESPACE)
{
}

ScatterAddPlugin::ScatterAddPlugin(const void* data, size_t length)
    : mNamespace(SCATTER_ADD_PLUGIN_NAMESPACE)
{
    const char* d = static_cast<const char*>(data);
    mAxis = *reinterpret_cast<const int*>(d);
}

const char* ScatterAddPlugin::getPluginType() const noexcept {
    return SCATTER_ADD_PLUGIN_NAME;
}

const char* ScatterAddPlugin::getPluginVersion() const noexcept {
    return SCATTER_ADD_PLUGIN_VERSION;
}

int ScatterAddPlugin::getNbOutputs() const noexcept {
    return 1;
}

int ScatterAddPlugin::initialize() noexcept {
    return 0;
}

void ScatterAddPlugin::terminate() noexcept {
}

size_t ScatterAddPlugin::getSerializationSize() const noexcept {
    return sizeof(mAxis);
}

void ScatterAddPlugin::serialize(void* buffer) const noexcept {
    char* d = static_cast<char*>(buffer);
    *reinterpret_cast<int*>(d) = mAxis;
}

void ScatterAddPlugin::destroy() noexcept {
    delete this;
}

void ScatterAddPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* ScatterAddPlugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

nvinfer1::DataType ScatterAddPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* inputTypes,
    int nbInputs
) const noexcept {
    // Output type matches data input type (input 0)
    return inputTypes[0];
}

nvinfer1::IPluginV2DynamicExt* ScatterAddPlugin::clone() const noexcept {
    auto* plugin = new ScatterAddPlugin(mAxis);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs ScatterAddPlugin::getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder
) noexcept {
    // Output has same shape as data input (input 0)
    // inputs[0] = data  [N, D]
    // inputs[1] = index [E]
    // inputs[2] = src   [E, D]
    return inputs[0];
}

bool ScatterAddPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs
) noexcept {
    assert(nbInputs == 3);
    assert(nbOutputs == 1);
    
    const auto& desc = inOut[pos];
    
    // All tensors must be in linear format
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }
    
    if (pos == 0) {
        // data: FP32 or FP16
        return desc.type == nvinfer1::DataType::kFLOAT || 
               desc.type == nvinfer1::DataType::kHALF;
    } else if (pos == 1) {
        // index: INT32 or INT64
        return desc.type == nvinfer1::DataType::kINT32 ||
               desc.type == nvinfer1::DataType::kINT64;
    } else if (pos == 2) {
        // src: must match data type
        return desc.type == inOut[0].type;
    } else {
        // output: must match data type
        return desc.type == inOut[0].type;
    }
}

void ScatterAddPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nbOutputs
) noexcept {
    // Nothing to configure
}

size_t ScatterAddPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nbOutputs
) const noexcept {
    // For mixed precision mode, we need a FP32 workspace
    if (inputs[0].type == nvinfer1::DataType::kHALF) {
        // data shape is [N, D]
        int N = inputs[0].dims.d[0];
        int D = inputs[0].dims.d[1];
        return getScatterAddWorkspaceSize(N, D);
    }
    return 0;
}

int ScatterAddPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream
) noexcept {
    // inputs[0] = data  [N, D]
    // inputs[1] = index [E]
    // inputs[2] = src   [E, D]
    
    const auto& dataDims = inputDesc[0].dims;
    const auto& indexDims = inputDesc[1].dims;
    const auto& srcDims = inputDesc[2].dims;
    
    int N = dataDims.d[0];
    int D = dataDims.d[1];
    int E = indexDims.d[0];
    
    // First, copy data to output (scatter_add accumulates on top of data)
    size_t dataBytes = N * D * (inputDesc[0].type == nvinfer1::DataType::kFLOAT ? 
                                sizeof(float) : sizeof(__half));
    cudaMemcpyAsync(outputs[0], inputs[0], dataBytes, cudaMemcpyDeviceToDevice, stream);
    
    if (E == 0) {
        return 0;  // No edges, just return data
    }
    
    // Handle different data types
    // Note: ONNX ScatterElements has indices shape [E, D] (expanded), not [E]
    // For axis=0, indices[e, d] are all the same in each row, so we use indices[e, 0]
    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT) {
        // FP32 path
        const float* src = static_cast<const float*>(inputs[2]);
        const int64_t* indices_2d = static_cast<const int64_t*>(inputs[1]);
        float* output = static_cast<float*>(outputs[0]);
        
        // Launch kernel that handles 2D expanded indices
        launchScatterAddFP32_2D(src, indices_2d, output, E, D, N, stream);
    } else {
        // FP16 path with mixed precision accumulation
        const __half* src = static_cast<const __half*>(inputs[2]);
        const int64_t* indices_2d = static_cast<const int64_t*>(inputs[1]);
        __half* output = static_cast<__half*>(outputs[0]);
        float* ws = static_cast<float*>(workspace);
        
        // For FP16, use mixed precision for numerical stability
        launchScatterAddFP16_2D(src, indices_2d, output, ws, E, D, N, stream);
    }
    
    return 0;
}

// ============================================================================
// ScatterAddPluginCreator Implementation
// ============================================================================

ScatterAddPluginCreator::ScatterAddPluginCreator()
    : mNamespace(SCATTER_ADD_PLUGIN_NAMESPACE)
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("axis", nullptr, nvinfer1::PluginFieldType::kINT32, 1)
    );
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ScatterAddPluginCreator::getPluginName() const noexcept {
    return SCATTER_ADD_PLUGIN_NAME;
}

const char* ScatterAddPluginCreator::getPluginVersion() const noexcept {
    return SCATTER_ADD_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* ScatterAddPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

nvinfer1::IPluginV2* ScatterAddPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc
) noexcept {
    int axis = 0;
    
    for (int i = 0; i < fc->nbFields; ++i) {
        const auto& field = fc->fields[i];
        if (strcmp(field.name, "axis") == 0) {
            axis = *static_cast<const int*>(field.data);
        }
    }
    
    return new ScatterAddPlugin(axis);
}

nvinfer1::IPluginV2* ScatterAddPluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength
) noexcept {
    return new ScatterAddPlugin(serialData, serialLength);
}

void ScatterAddPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* ScatterAddPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

}  // namespace plugin
}  // namespace planr1

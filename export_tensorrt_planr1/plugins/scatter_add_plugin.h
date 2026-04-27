/*
 * ScatterAdd TensorRT Plugin
 * 
 * Implements scatter_add operation as a TensorRT plugin.
 * This replaces ONNX ScatterElements with reduction='add'.
 */

#pragma once

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <string>
#include <vector>

namespace planr1 {
namespace plugin {

// Plugin version and name
static const char* SCATTER_ADD_PLUGIN_VERSION = "1";
static const char* SCATTER_ADD_PLUGIN_NAME = "ScatterAdd";
static const char* SCATTER_ADD_PLUGIN_NAMESPACE = "";  // Use default namespace

/**
 * ScatterAdd Plugin
 * 
 * Inputs:
 *   0: data   [N, D] - base tensor (usually zeros)
 *   1: index  [E]    - scatter indices
 *   2: src    [E, D] - source values to scatter
 * 
 * Output:
 *   0: output [N, D] - result of scatter_add
 * 
 * Operation:
 *   output = data.clone()
 *   output[index[e], :] += src[e, :] for all e
 */
class ScatterAddPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    ScatterAddPlugin(int axis = 0);
    ScatterAddPlugin(const void* data, size_t length);
    ~ScatterAddPlugin() override = default;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(
        int index,
        const nvinfer1::DataType* inputTypes,
        int nbInputs
    ) const noexcept override;

    // IPluginV2DynamicExt methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder
    ) noexcept override;
    
    bool supportsFormatCombination(
        int pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int nbInputs,
        int nbOutputs
    ) noexcept override;
    
    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in,
        int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out,
        int nbOutputs
    ) noexcept override;
    
    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs,
        int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int nbOutputs
    ) const noexcept override;
    
    int enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream
    ) noexcept override;

private:
    int mAxis;
    std::string mNamespace;
};

/**
 * ScatterAdd Plugin Creator
 */
class ScatterAddPluginCreator : public nvinfer1::IPluginCreator {
public:
    ScatterAddPluginCreator();
    ~ScatterAddPluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    
    nvinfer1::IPluginV2* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* fc
    ) noexcept override;
    
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t serialLength
    ) noexcept override;
    
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

// Register plugin
REGISTER_TENSORRT_PLUGIN(ScatterAddPluginCreator);

}  // namespace plugin
}  // namespace planr1

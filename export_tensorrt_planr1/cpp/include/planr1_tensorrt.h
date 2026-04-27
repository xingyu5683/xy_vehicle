/*
 * Plan-R1 TensorRT Inference Header
 * 
 * High-performance inference using TensorRT FP16.
 */

#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace planr1 {

// ============================================================================
// Constants
// ============================================================================

constexpr int MAX_AGENTS = 21;
constexpr int MAX_POLYGONS = 145;
constexpr int MAX_POLYLINES = 1200;
constexpr int NUM_TOKENS = 1024;
constexpr int NUM_FUTURE_STEPS = 16;
constexpr int EMBED_DIM = 128;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Map encoder input data
 */
struct MapEncoderInput {
    std::vector<float> polyline_position;      // [MAX_POLYLINES, 2]
    std::vector<float> polyline_heading;       // [MAX_POLYLINES]
    std::vector<float> polyline_length;        // [MAX_POLYLINES]
    std::vector<float> polygon_position;       // [MAX_POLYGONS, 2]
    std::vector<float> polygon_heading;        // [MAX_POLYGONS]
    std::vector<float> polygon_speed_limit;    // [MAX_POLYGONS]
    std::vector<float> polygon_speed_limit_valid; // [MAX_POLYGONS]
    std::vector<int64_t> polygon_type;         // [MAX_POLYGONS]
    std::vector<int64_t> polygon_traffic_light;// [MAX_POLYGONS]
    std::vector<int64_t> polygon_on_route;     // [MAX_POLYGONS]
    std::vector<int64_t> l2g_edge_index;       // [2, MAX_POLYLINES]
    std::vector<int64_t> left_edge_index;      // [2, MAX_EDGES]
    std::vector<int64_t> right_edge_index;     // [2, MAX_EDGES]
    std::vector<int64_t> incoming_edge_index;  // [2, MAX_EDGES]
    std::vector<int64_t> outgoing_edge_index;  // [2, MAX_EDGES]
};

/**
 * Step model input data
 */
struct StepModelInput {
    std::vector<int64_t> agent_token;          // [MAX_AGENTS, num_intervals]
    std::vector<float> agent_position;         // [MAX_AGENTS, num_intervals, 2]
    std::vector<float> agent_heading;          // [MAX_AGENTS, num_intervals]
    std::vector<float> agent_velocity;         // [MAX_AGENTS, num_intervals, 2]
    std::vector<int64_t> agent_type;           // [MAX_AGENTS]
    std::vector<float> agent_valid_mask;       // [MAX_AGENTS, num_intervals]
    std::vector<float> polygon_embs;           // [MAX_POLYGONS, EMBED_DIM]
    
    // Dynamic edges
    std::vector<int64_t> k2k_t_edge_index;     // [2, E1]
    std::vector<float> k2k_t_edge_attr;        // [E1, 6]
    std::vector<int64_t> g2k_edge_index;       // [2, E2]
    std::vector<float> g2k_edge_attr;          // [E2, 6]
    std::vector<int64_t> k2k_a_edge_index;     // [2, E3]
    std::vector<float> k2k_a_edge_attr;        // [E3, 5]
    
    int num_agents = MAX_AGENTS;
    int num_intervals = 5;
};

/**
 * Inference output
 */
struct InferenceOutput {
    std::vector<float> trajectories;           // [MAX_AGENTS, NUM_FUTURE_STEPS, 3]
    std::vector<int64_t> predicted_tokens;     // [MAX_AGENTS, NUM_FUTURE_STEPS]
    float map_encoder_time_ms;
    float step_time_ms;
    float total_time_ms;
};

/**
 * Token dictionary entry: (delta_x, delta_y, delta_heading)
 */
using TokenDict = std::vector<std::vector<std::array<float, 3>>>;

// ============================================================================
// TensorRT Logger
// ============================================================================

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
    void setVerbosity(Severity severity) { mSeverity = severity; }
    
private:
    Severity mSeverity = Severity::kWARNING;
};

// ============================================================================
// PlanR1TensorRT Class
// ============================================================================

class PlanR1TensorRT {
public:
    /**
     * Constructor
     * 
     * @param map_encoder_path Path to map_encoder.trt
     * @param step_model_path Path to step.trt
     * @param token_dict_path Path to tokens_1024.bin
     * @param plugin_path Path to libscatter_add_plugin.so (optional)
     */
    PlanR1TensorRT(
        const std::string& map_encoder_path,
        const std::string& step_model_path,
        const std::string& token_dict_path,
        const std::string& plugin_path = ""
    );
    
    ~PlanR1TensorRT();
    
    // Disable copy
    PlanR1TensorRT(const PlanR1TensorRT&) = delete;
    PlanR1TensorRT& operator=(const PlanR1TensorRT&) = delete;
    
    /**
     * Run map encoder
     * @return polygon embeddings [MAX_POLYGONS, EMBED_DIM]
     */
    std::vector<float> runMapEncoder(const MapEncoderInput& input);
    
    /**
     * Run single step inference
     * @return logits [MAX_AGENTS, NUM_TOKENS]
     */
    std::vector<float> runSingleStep(const StepModelInput& input);
    
    /**
     * Run full autoregressive inference (16 steps)
     */
    InferenceOutput runInference(StepModelInput& input);
    
    /**
     * Get timing statistics
     */
    float getLastMapEncoderTimeMs() const { return mLastMapEncoderTime; }
    float getLastStepTimeMs() const { return mLastStepTime; }
    
private:
    // Load TensorRT engine from file
    nvinfer1::ICudaEngine* loadEngine(const std::string& path);
    
    // Load token dictionary
    TokenDict loadTokenDict(const std::string& path);
    
    // Decode token to motion delta
    std::array<float, 3> decodeToken(int64_t token, int64_t agent_type);
    
    // Allocate GPU buffers
    void allocateBuffers();
    void freeBuffers();
    
private:
    TRTLogger mLogger;
    
    // TensorRT components
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mMapEncoderEngine;
    std::unique_ptr<nvinfer1::ICudaEngine> mStepEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mMapEncoderContext;
    std::unique_ptr<nvinfer1::IExecutionContext> mStepContext;
    
    // CUDA stream
    cudaStream_t mStream;
    
    // Token dictionary
    TokenDict mTokenDict;
    
    // GPU buffers (pre-allocated for performance)
    std::unordered_map<std::string, void*> mDeviceBuffers;
    std::unordered_map<std::string, size_t> mBufferSizes;
    
    // Timing
    float mLastMapEncoderTime = 0.0f;
    float mLastStepTime = 0.0f;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Load token dictionary from binary file
 */
TokenDict loadTokenDictionary(const std::string& path);

/**
 * Compute edge information for step model
 */
void computeEdges(
    const std::vector<float>& agent_positions,
    const std::vector<float>& agent_headings,
    const std::vector<bool>& agent_valid_mask,
    const std::vector<float>& polygon_positions,
    const std::vector<float>& polygon_headings,
    int num_agents,
    int num_polygons,
    int num_intervals,
    float agent_radius,
    float polygon_radius,
    int interval_frames,
    std::vector<int64_t>& k2k_t_edge_index,
    std::vector<float>& k2k_t_edge_attr,
    std::vector<int64_t>& g2k_edge_index,
    std::vector<float>& g2k_edge_attr,
    std::vector<int64_t>& k2k_a_edge_index,
    std::vector<float>& k2k_a_edge_attr
);

}  // namespace planr1

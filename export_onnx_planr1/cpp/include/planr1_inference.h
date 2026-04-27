/**
 * Plan-R1 ONNX Inference Header
 * 
 * C++ inference loop for Plan-R1 model using ONNX Runtime.
 * 
 * Architecture:
 *   1. map_encoder.onnx: Computes polygon embeddings (run once per scene)
 *   2. step.onnx: Single inference step (run 16 times in autoregressive loop)
 *   External C++ loop handles the autoregressive iteration.
 */

#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <array>
#include <unordered_map>

namespace planr1 {

// ============================================================
// Constants
// ============================================================

constexpr int MAX_AGENTS = 21;          // 20 neighbors + 1 ego
constexpr int MAX_POLYGONS = 145;
constexpr int MAX_POLYLINES = 1200;
constexpr int MAX_EDGES = 80;
constexpr int HIDDEN_DIM = 128;
constexpr int NUM_TOKENS = 1024;
constexpr int NUM_FUTURE_STEPS = 16;    // 16 steps * 0.5s = 8s trajectory
constexpr int INTERVAL = 5;             // 5 frames per token

// ============================================================
// Data Structures
// ============================================================

/**
 * Token dictionary entry for decoding motion tokens.
 * Each token represents a motion primitive (delta_x, delta_y, delta_heading).
 */
struct TokenEntry {
    float delta_x;      // Position change in x
    float delta_y;      // Position change in y  
    float delta_heading; // Heading change
};

/**
 * Agent state for tracking during autoregressive inference.
 */
struct AgentState {
    float position_x;
    float position_y;
    float heading;
    int current_token;
    bool is_valid;
};

/**
 * Map encoder input data.
 */
struct MapEncoderInput {
    // Polyline features [MAX_POLYLINES, ...]
    std::vector<float> polyline_position;      // [1200, 2]
    std::vector<float> polyline_heading;       // [1200]
    std::vector<float> polyline_length;        // [1200]
    
    // Polygon features [MAX_POLYGONS, ...]
    std::vector<float> polygon_position;       // [145, 2]
    std::vector<float> polygon_heading;        // [145]
    std::vector<int64_t> polygon_heading_valid; // [145]
    std::vector<int64_t> polygon_type;         // [145]
    std::vector<int64_t> polygon_traffic_light; // [145]
    std::vector<float> polygon_speed_limit;    // [145]
    std::vector<float> polygon_speed_limit_valid_float; // [145] - FLOAT in ONNX
    std::vector<int64_t> polygon_on_route;     // [145]
    
    // Topology edges (all [2, 80] for consistency)
    std::vector<int64_t> left_edge_index;      // [2, 80]
    std::vector<int64_t> right_edge_index;     // [2, 80]
    std::vector<int64_t> incoming_edge_index;  // [2, 80]
    std::vector<int64_t> outgoing_edge_index;  // [2, 80]
    std::vector<int64_t> polyline_to_polygon_edge_index; // [2, 1200] - l2g_edge_index
    
    int num_polylines;
    int num_polygons;
};

/**
 * Step model input data for a single inference step.
 * Matches the actual ONNX model inputs.
 */
struct StepModelInput {
    // Agent features
    std::vector<int64_t> agent_token;          // [21, num_intervals] - token history
    std::vector<int64_t> agent_type;           // [21]
    std::vector<float> agent_box;              // [21, 4]
    std::vector<int64_t> agent_identity;       // [21]
    
    // Pre-computed embeddings from map_encoder
    std::vector<float> polygon_embs;           // [145, 128]
    
    // Edge data (pre-computed, fixed for all 16 steps)
    std::vector<int64_t> k2k_t_edge_index;     // [2, num_k2k_t_edges]
    std::vector<float> k2k_t_edge_attr;        // [num_k2k_t_edges, 6]
    std::vector<int64_t> g2k_edge_index;       // [2, num_g2k_edges]
    std::vector<float> g2k_edge_attr;          // [num_g2k_edges, 6]
    std::vector<int64_t> k2k_a_edge_index;     // [2, num_k2k_a_edges]
    std::vector<float> k2k_a_edge_attr;        // [num_k2k_a_edges, 5]
    
    // Actual counts
    int num_agents;
    int num_intervals;  // Current T (hist + predicted so far)
    int num_k2k_t_edges;
    int num_g2k_edges;
    int num_k2k_a_edges;
};

/**
 * Inference output: predicted trajectory for all agents.
 */
struct InferenceOutput {
    // Output trajectories [num_agents, 16, 2] and [num_agents, 16]
    std::vector<float> positions;   // [num_agents * 16 * 2]
    std::vector<float> headings;    // [num_agents * 16]
    int num_agents;
};

// ============================================================
// PlanR1 Inference Engine
// ============================================================

class PlanR1Inference {
public:
    /**
     * Constructor.
     * @param map_encoder_path Path to map_encoder.onnx
     * @param step_model_path Path to step.onnx
     * @param token_dict Token dictionary for decoding
     * @param use_gpu Whether to use GPU (CUDA)
     */
    PlanR1Inference(
        const std::string& map_encoder_path,
        const std::string& step_model_path,
        const std::unordered_map<int, TokenEntry>& token_dict,
        bool use_gpu = true
    );
    
    ~PlanR1Inference();
    
    /**
     * Run map encoder to compute polygon embeddings.
     * Call this once per scene.
     * 
     * @param input Map encoder input data
     * @return Polygon embeddings [MAX_POLYGONS, HIDDEN_DIM]
     */
    std::vector<float> runMapEncoder(const MapEncoderInput& input);
    
    /**
     * Run full autoregressive inference.
     * 
     * @param input Initial step model input (with polygon_embs from runMapEncoder)
     * @return Predicted trajectories for all agents
     */
    InferenceOutput runInference(StepModelInput& input);
    
    /**
     * Run single step inference.
     * 
     * @param input Step model input
     * @return Logits [num_agents, NUM_TOKENS]
     */
    std::vector<float> runSingleStep(const StepModelInput& input);
    
    /**
     * Decode token to trajectory delta.
     * 
     * @param token Token ID
     * @param agent_type Agent type (0=Vehicle, 1=Pedestrian, 2=Bicycle)
     * @return TokenEntry with delta_x, delta_y, delta_heading
     */
    TokenEntry decodeToken(int token, int agent_type) const;
    
private:
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> map_encoder_session_;
    std::unique_ptr<Ort::Session> step_model_session_;
    Ort::MemoryInfo memory_info_;
    
    // Token dictionary: agent_type -> token_id -> TokenEntry
    std::unordered_map<int, std::unordered_map<int, TokenEntry>> token_dict_;
    
    // Helper functions
    void setupSessionOptions(bool use_gpu);
    std::vector<float> createInputTensor(const std::vector<float>& data, 
                                         const std::vector<int64_t>& shape);
    int argmax(const std::vector<float>& logits, int offset, int size);
};

// ============================================================
// Utility Functions
// ============================================================

/**
 * Load token dictionary from file.
 * Expected format: binary file with token entries.
 */
std::unordered_map<int, std::unordered_map<int, TokenEntry>> 
loadTokenDictionary(const std::string& path);

/**
 * Compute edges for attention layers.
 * This should be called once during preprocessing.
 * 
 * Edge index formats:
 *   - k2k_t, g2k: agent_major index = agent * num_intervals + timestep
 *   - k2k_a: time_major index = timestep * MAX_AGENTS + agent
 * 
 * Edge attributes:
 *   - k2k_t: [length, cos(theta), sin(theta), cos(heading), sin(heading), interval]
 *   - g2k:   [length, cos(theta), sin(theta), cos(heading), sin(heading), heading_valid]
 *   - k2k_a: [length, cos(theta), sin(theta), cos(heading), sin(heading)]
 */
void computeEdges(
    const std::vector<float>& agent_positions,   // [num_agents * num_intervals * 2] - 每个时间步的位置
    const std::vector<float>& agent_headings,    // [num_agents * num_intervals] - 每个时间步的航向
    const std::vector<bool>& agent_valid_mask,   // [num_agents * num_intervals] - 有效性
    const std::vector<float>& polygon_positions, // [num_polygons * 2]
    const std::vector<float>& polygon_headings,  // [num_polygons]
    int num_agents,
    int num_polygons,
    int num_intervals,
    float agent_radius,      // 通常为 60.0
    float polygon_radius,    // 通常为 30.0
    int interval_frames,     // 每个 interval 的帧数 (通常为 5)
    // Outputs
    std::vector<int64_t>& k2k_t_edge_index,
    std::vector<float>& k2k_t_edge_attr,
    std::vector<int64_t>& g2k_edge_index,
    std::vector<float>& g2k_edge_attr,
    std::vector<int64_t>& k2k_a_edge_index,
    std::vector<float>& k2k_a_edge_attr
);

/**
 * Wrap angle to [-pi, pi].
 */
inline float wrapAngle(float angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

} // namespace planr1

/**
 * Plan-R1 ONNX Inference Implementation
 * 
 * C++ inference loop for Plan-R1 model using ONNX Runtime.
 */

#include "planr1_inference.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdexcept>

namespace planr1 {

// ============================================================
// PlanR1Inference Implementation
// ============================================================

PlanR1Inference::PlanR1Inference(
    const std::string& map_encoder_path,
    const std::string& step_model_path,
    const std::unordered_map<int, TokenEntry>& token_dict,
    bool use_gpu
) : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    
    // Initialize ONNX Runtime environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PlanR1Inference");
    
    // Setup session options
    session_options_ = std::make_unique<Ort::SessionOptions>();
    setupSessionOptions(use_gpu);
    
    // Load models
    std::cout << "Loading map_encoder from: " << map_encoder_path << std::endl;
    map_encoder_session_ = std::make_unique<Ort::Session>(*env_, map_encoder_path.c_str(), *session_options_);
    
    std::cout << "Loading step_model from: " << step_model_path << std::endl;
    step_model_session_ = std::make_unique<Ort::Session>(*env_, step_model_path.c_str(), *session_options_);
    
    // Store token dictionary
    // Convert flat dict to nested dict by agent type
    for (const auto& [token_id, entry] : token_dict) {
        // Assuming token_dict is already organized by type
        // For now, store in type 0 (Vehicle)
        token_dict_[0][token_id] = entry;
        token_dict_[1][token_id] = entry;  // Pedestrian
        token_dict_[2][token_id] = entry;  // Bicycle
    }
    
    std::cout << "PlanR1Inference initialized successfully" << std::endl;
}

PlanR1Inference::~PlanR1Inference() = default;

void PlanR1Inference::setupSessionOptions(bool use_gpu) {
    session_options_->SetIntraOpNumThreads(4);
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    if (use_gpu) {
#ifdef USE_CUDA
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options_->AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "Using CUDA execution provider" << std::endl;
#else
        std::cout << "CUDA not available, using CPU" << std::endl;
#endif
    }
}

std::vector<float> PlanR1Inference::runMapEncoder(const MapEncoderInput& input) {
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Get input names from session
    size_t num_inputs = map_encoder_session_->GetInputCount();
    std::vector<std::string> input_name_strings;
    std::cout << "  MapEncoder expects " << num_inputs << " inputs:" << std::endl;
    for (size_t i = 0; i < num_inputs; i++) {
        auto name = map_encoder_session_->GetInputNameAllocated(i, allocator);
        input_name_strings.push_back(name.get());
        std::cout << "    " << i << ": " << name.get() << std::endl;
    }
    
    // Prepare input tensors in the exact order expected by ONNX model
    std::vector<Ort::Value> input_tensors;
    
    // Shape definitions
    std::vector<int64_t> polyline_pos_shape = {MAX_POLYLINES, 2};
    std::vector<int64_t> polyline_shape = {MAX_POLYLINES};
    std::vector<int64_t> polygon_pos_shape = {MAX_POLYGONS, 2};
    std::vector<int64_t> polygon_shape = {MAX_POLYGONS};
    std::vector<int64_t> l2g_edge_shape = {2, MAX_POLYLINES};
    std::vector<int64_t> edge_80_shape = {2, 80};
    
    // 1. polyline_position [1200, 2]
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(input.polyline_position.data()),
        input.polyline_position.size(), polyline_pos_shape.data(), polyline_pos_shape.size()));
    
    // 2. polyline_heading [1200]
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(input.polyline_heading.data()),
        input.polyline_heading.size(), polyline_shape.data(), polyline_shape.size()));
    
    // 3. polyline_length [1200]
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(input.polyline_length.data()),
        input.polyline_length.size(), polyline_shape.data(), polyline_shape.size()));
    
    // 4. polygon_position [145, 2]
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(input.polygon_position.data()),
        input.polygon_position.size(), polygon_pos_shape.data(), polygon_pos_shape.size()));
    
    // 5. polygon_heading [145]
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(input.polygon_heading.data()),
        input.polygon_heading.size(), polygon_shape.data(), polygon_shape.size()));
    
    // 6. polygon_speed_limit [145]
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(input.polygon_speed_limit.data()),
        input.polygon_speed_limit.size(), polygon_shape.data(), polygon_shape.size()));
    
    // 7. polygon_speed_limit_valid [145] - FLOAT type
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(input.polygon_speed_limit_valid_float.data()),
        input.polygon_speed_limit_valid_float.size(), polygon_shape.data(), polygon_shape.size()));
    
    // 8. polygon_type [145]
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t*>(input.polygon_type.data()),
        input.polygon_type.size(), polygon_shape.data(), polygon_shape.size()));
    
    // 9. polygon_traffic_light [145]
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t*>(input.polygon_traffic_light.data()),
        input.polygon_traffic_light.size(), polygon_shape.data(), polygon_shape.size()));
    
    // 10. polygon_on_route [145]
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t*>(input.polygon_on_route.data()),
        input.polygon_on_route.size(), polygon_shape.data(), polygon_shape.size()));
    
    // 11. l2g_edge_index [2, 1200] - polyline to polygon edge
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t*>(input.polyline_to_polygon_edge_index.data()),
        input.polyline_to_polygon_edge_index.size(), l2g_edge_shape.data(), l2g_edge_shape.size()));
    
    // 12. left_edge_index [2, 80]
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t*>(input.left_edge_index.data()),
        input.left_edge_index.size(), edge_80_shape.data(), edge_80_shape.size()));
    
    // 13. right_edge_index [2, 80]
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t*>(input.right_edge_index.data()),
        input.right_edge_index.size(), edge_80_shape.data(), edge_80_shape.size()));
    
    // 14. incoming_edge_index [2, 80]
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t*>(input.incoming_edge_index.data()),
        input.incoming_edge_index.size(), edge_80_shape.data(), edge_80_shape.size()));
    
    // 15. outgoing_edge_index [2, 80]
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t*>(input.outgoing_edge_index.data()),
        input.outgoing_edge_index.size(), edge_80_shape.data(), edge_80_shape.size()));
    
    // Prepare input/output names
    std::vector<const char*> input_names_cstr;
    for (const auto& name : input_name_strings) {
        input_names_cstr.push_back(name.c_str());
    }
    
    // Get output names
    size_t num_outputs = map_encoder_session_->GetOutputCount();
    std::vector<std::string> output_name_strings;
    for (size_t i = 0; i < num_outputs; i++) {
        auto name = map_encoder_session_->GetOutputNameAllocated(i, allocator);
        output_name_strings.push_back(name.get());
    }
    std::vector<const char*> output_names_cstr;
    for (const auto& name : output_name_strings) {
        output_names_cstr.push_back(name.c_str());
    }
    
    // Run inference
    auto output_tensors = map_encoder_session_->Run(
        Ort::RunOptions{nullptr},
        input_names_cstr.data(), input_tensors.data(), input_tensors.size(),
        output_names_cstr.data(), output_names_cstr.size()
    );
    
    // Extract output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }
    
    return std::vector<float>(output_data, output_data + output_size);
}

std::vector<float> PlanR1Inference::runSingleStep(const StepModelInput& input) {
    Ort::AllocatorWithDefaultOptions allocator;
    
    std::vector<Ort::Value> input_tensors;
    
    // Get input names from ONNX model
    size_t num_inputs = step_model_session_->GetInputCount();
    std::vector<std::string> input_name_strings;
    for (size_t i = 0; i < num_inputs; i++) {
        auto name = step_model_session_->GetInputNameAllocated(i, allocator);
        input_name_strings.push_back(name.get());
    }
    
    // ONNX Model expects these 11 inputs in order:
    // 1. agent_token: INT64 [21, num_intervals]
    // 2. agent_type: INT64 [21]
    // 3. agent_box: FLOAT [21, 4]
    // 4. agent_identity: INT64 [21]
    // 5. polygon_embs: FLOAT [145, 128]
    // 6. k2k_t_edge_index: INT64 [2, num_k2k_t_edges]
    // 7. k2k_t_edge_attr: FLOAT [num_k2k_t_edges, 6]
    // 8. g2k_edge_index: INT64 [2, num_g2k_edges]
    // 9. g2k_edge_attr: FLOAT [num_g2k_edges, 6]
    // 10. k2k_a_edge_index: INT64 [2, num_k2k_a_edges]
    // 11. k2k_a_edge_attr: FLOAT [num_k2k_a_edges, 5]
    
    // 1. agent_token [21, num_intervals]
    std::vector<int64_t> agent_token_shape = {MAX_AGENTS, (int64_t)input.num_intervals};
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t*>(input.agent_token.data()),
        input.agent_token.size(), agent_token_shape.data(), agent_token_shape.size()));
    
    // 2. agent_type [21]
    std::vector<int64_t> agent_shape = {MAX_AGENTS};
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t*>(input.agent_type.data()),
        input.agent_type.size(), agent_shape.data(), agent_shape.size()));
    
    // 3. agent_box [21, 4]
    std::vector<int64_t> agent_box_shape = {MAX_AGENTS, 4};
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(input.agent_box.data()),
        input.agent_box.size(), agent_box_shape.data(), agent_box_shape.size()));
    
    // 4. agent_identity [21]
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t*>(input.agent_identity.data()),
        input.agent_identity.size(), agent_shape.data(), agent_shape.size()));
    
    // 5. polygon_embs [145, 128]
    std::vector<int64_t> polygon_embs_shape = {MAX_POLYGONS, HIDDEN_DIM};
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(input.polygon_embs.data()),
        input.polygon_embs.size(), polygon_embs_shape.data(), polygon_embs_shape.size()));
    
    // 6. k2k_t_edge_index [2, num_k2k_t_edges]
    std::vector<int64_t> k2k_t_edge_shape = {2, (int64_t)input.num_k2k_t_edges};
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t*>(input.k2k_t_edge_index.data()),
        input.k2k_t_edge_index.size(), k2k_t_edge_shape.data(), k2k_t_edge_shape.size()));
    
    // 7. k2k_t_edge_attr [num_k2k_t_edges, 6]
    std::vector<int64_t> k2k_t_attr_shape = {(int64_t)input.num_k2k_t_edges, 6};
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(input.k2k_t_edge_attr.data()),
        input.k2k_t_edge_attr.size(), k2k_t_attr_shape.data(), k2k_t_attr_shape.size()));
    
    // 8. g2k_edge_index [2, num_g2k_edges]
    std::vector<int64_t> g2k_edge_shape = {2, (int64_t)input.num_g2k_edges};
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t*>(input.g2k_edge_index.data()),
        input.g2k_edge_index.size(), g2k_edge_shape.data(), g2k_edge_shape.size()));
    
    // 9. g2k_edge_attr [num_g2k_edges, 6]
    std::vector<int64_t> g2k_attr_shape = {(int64_t)input.num_g2k_edges, 6};
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(input.g2k_edge_attr.data()),
        input.g2k_edge_attr.size(), g2k_attr_shape.data(), g2k_attr_shape.size()));
    
    // 10. k2k_a_edge_index [2, num_k2k_a_edges]
    std::vector<int64_t> k2k_a_edge_shape = {2, (int64_t)input.num_k2k_a_edges};
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t*>(input.k2k_a_edge_index.data()),
        input.k2k_a_edge_index.size(), k2k_a_edge_shape.data(), k2k_a_edge_shape.size()));
    
    // 11. k2k_a_edge_attr [num_k2k_a_edges, 5]
    std::vector<int64_t> k2k_a_attr_shape = {(int64_t)input.num_k2k_a_edges, 5};
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(input.k2k_a_edge_attr.data()),
        input.k2k_a_edge_attr.size(), k2k_a_attr_shape.data(), k2k_a_attr_shape.size()));
    
    // Prepare names
    std::vector<const char*> input_names_cstr;
    for (const auto& name : input_name_strings) {
        input_names_cstr.push_back(name.c_str());
    }
    
    size_t num_outputs = step_model_session_->GetOutputCount();
    std::vector<std::string> output_name_strings;
    for (size_t i = 0; i < num_outputs; i++) {
        auto name = step_model_session_->GetOutputNameAllocated(i, allocator);
        output_name_strings.push_back(name.get());
    }
    std::vector<const char*> output_names_cstr;
    for (const auto& name : output_name_strings) {
        output_names_cstr.push_back(name.c_str());
    }
    
    // Run inference
    auto output_tensors = step_model_session_->Run(
        Ort::RunOptions{nullptr},
        input_names_cstr.data(), input_tensors.data(), input_tensors.size(),
        output_names_cstr.data(), output_names_cstr.size()
    );
    
    // Extract logits [num_agents, NUM_TOKENS]
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }
    
    return std::vector<float>(output_data, output_data + output_size);
}

InferenceOutput PlanR1Inference::runInference(StepModelInput& input) {
    InferenceOutput output;
    output.num_agents = input.num_agents;
    output.positions.resize(input.num_agents * NUM_FUTURE_STEPS * 2);
    output.headings.resize(input.num_agents * NUM_FUTURE_STEPS);
    
    // Track current positions and headings for each agent
    // Note: Initial positions should be provided externally
    std::vector<float> current_positions(input.num_agents * 2, 0.0f);
    std::vector<float> current_headings(input.num_agents, 0.0f);
    
    // Autoregressive loop
    for (int step = 0; step < NUM_FUTURE_STEPS; step++) {
        // Run single step inference
        std::vector<float> logits = runSingleStep(input);
        
        // For each agent, sample next token and decode
        for (int a = 0; a < input.num_agents; a++) {
            // Argmax to get next token
            int next_token = argmax(logits, a * NUM_TOKENS, NUM_TOKENS);
            
            // Get agent type
            int agent_type = static_cast<int>(input.agent_type[a]);
            
            // Decode token to delta
            TokenEntry delta = decodeToken(next_token, agent_type);
            
            // Get current heading for rotation
            float heading = current_headings[a];
            float cos_h = std::cos(heading);
            float sin_h = std::sin(heading);
            
            // Rotate delta by current heading (local to global)
            float dx_global = delta.delta_x * cos_h - delta.delta_y * sin_h;
            float dy_global = delta.delta_x * sin_h + delta.delta_y * cos_h;
            
            // Update position
            current_positions[a * 2] += dx_global;
            current_positions[a * 2 + 1] += dy_global;
            current_headings[a] = wrapAngle(current_headings[a] + delta.delta_heading);
            
            // Store in output
            int out_idx = a * NUM_FUTURE_STEPS * 2 + step * 2;
            output.positions[out_idx] = current_positions[a * 2];
            output.positions[out_idx + 1] = current_positions[a * 2 + 1];
            output.headings[a * NUM_FUTURE_STEPS + step] = current_headings[a];
            
            // Update token for next step
            input.agent_token[a] = next_token;
        }
        
        // Update token history for next step
        // Expand agent_token to include the new token
        int old_intervals = input.num_intervals;
        input.num_intervals++;
        
        // Resize agent_token array: [21, old_intervals] -> [21, new_intervals]
        std::vector<int64_t> new_agent_token(MAX_AGENTS * input.num_intervals, 0);
        for (int a = 0; a < MAX_AGENTS; a++) {
            // Copy old tokens
            for (int t = 0; t < old_intervals; t++) {
                new_agent_token[a * input.num_intervals + t] = 
                    input.agent_token[a * old_intervals + t];
            }
            // Add new token
            if (a < input.num_agents) {
                int next_token = argmax(logits, a * NUM_TOKENS, NUM_TOKENS);
                new_agent_token[a * input.num_intervals + old_intervals] = next_token;
            }
        }
        input.agent_token = std::move(new_agent_token);
    }
    
    return output;
}

TokenEntry PlanR1Inference::decodeToken(int token, int agent_type) const {
    // Clamp agent_type to valid range
    agent_type = std::clamp(agent_type, 0, 2);
    
    auto type_it = token_dict_.find(agent_type);
    if (type_it == token_dict_.end()) {
        // Fallback to zero delta
        return {0.0f, 0.0f, 0.0f};
    }
    
    auto token_it = type_it->second.find(token);
    if (token_it == type_it->second.end()) {
        // Unknown token, return zero delta
        return {0.0f, 0.0f, 0.0f};
    }
    
    return token_it->second;
}

int PlanR1Inference::argmax(const std::vector<float>& logits, int offset, int size) {
    int max_idx = 0;
    float max_val = logits[offset];
    
    for (int i = 1; i < size; i++) {
        if (logits[offset + i] > max_val) {
            max_val = logits[offset + i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

// ============================================================
// Utility Functions
// ============================================================

void computeEdges(
    const std::vector<float>& agent_positions,   // [num_agents * num_intervals, 2] - 每个时间步的位置
    const std::vector<float>& agent_headings,    // [num_agents * num_intervals] - 每个时间步的航向
    const std::vector<bool>& agent_valid_mask,   // [num_agents * num_intervals] - 有效性
    const std::vector<float>& polygon_positions, // [num_polygons, 2]
    const std::vector<float>& polygon_headings,  // [num_polygons]
    int num_agents,
    int num_polygons,
    int num_intervals,
    float agent_radius,
    float polygon_radius,
    int interval_frames,  // 每个 interval 的帧数 (通常为 5)
    std::vector<int64_t>& k2k_t_edge_index,
    std::vector<float>& k2k_t_edge_attr,
    std::vector<int64_t>& g2k_edge_index,
    std::vector<float>& g2k_edge_attr,
    std::vector<int64_t>& k2k_a_edge_index,
    std::vector<float>& k2k_a_edge_attr
) {
    // Clear outputs
    k2k_t_edge_index.clear();
    k2k_t_edge_attr.clear();
    g2k_edge_index.clear();
    g2k_edge_attr.clear();
    k2k_a_edge_index.clear();
    k2k_a_edge_attr.clear();
    
    std::vector<int64_t> k2k_t_edges_src, k2k_t_edges_dst;
    std::vector<int64_t> g2k_edges_src, g2k_edges_dst;
    std::vector<int64_t> k2k_a_edges_src, k2k_a_edges_dst;
    
    // Helper: transform point to local coordinate
    auto transformToLocal = [](float dx, float dy, float heading) -> std::pair<float, float> {
        float cos_h = std::cos(-heading);
        float sin_h = std::sin(-heading);
        return {dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h};
    };
    
    // ========================================
    // k2k_t: Temporal self-attention edges
    // Index format: agent_major -> idx = agent * num_intervals + timestep
    // Only connect t_src to t_dst where t_src <= t_dst and t_dst - t_src <= 6
    // ========================================
    for (int a = 0; a < num_agents; a++) {
        for (int t_dst = 0; t_dst < num_intervals; t_dst++) {
            int dst_idx = a * num_intervals + t_dst;
            if (!agent_valid_mask[dst_idx]) continue;
            
            float dst_x = agent_positions[dst_idx * 2];
            float dst_y = agent_positions[dst_idx * 2 + 1];
            float dst_h = agent_headings[dst_idx];
            
            for (int t_src = std::max(0, t_dst - 6); t_src <= t_dst; t_src++) {
                int src_idx = a * num_intervals + t_src;
                if (!agent_valid_mask[src_idx]) continue;
                
                float src_x = agent_positions[src_idx * 2];
                float src_y = agent_positions[src_idx * 2 + 1];
                float src_h = agent_headings[src_idx];
                
                // Transform src position to dst local coordinate
                float dx = src_x - dst_x;
                float dy = src_y - dst_y;
                auto [local_x, local_y] = transformToLocal(dx, dy, dst_h);
                float length = std::sqrt(local_x * local_x + local_y * local_y) + 1e-8f;
                float theta = std::atan2(local_y, local_x);
                float heading_diff = wrapAngle(src_h - dst_h);
                float dt = static_cast<float>(t_src - t_dst) * interval_frames;
                
                k2k_t_edges_src.push_back(src_idx);
                k2k_t_edges_dst.push_back(dst_idx);
                
                // Edge attr: [length, cos(theta), sin(theta), cos(heading), sin(heading), interval]
                k2k_t_edge_attr.insert(k2k_t_edge_attr.end(), {
                    length, std::cos(theta), std::sin(theta),
                    std::cos(heading_diff), std::sin(heading_diff), dt
                });
            }
        }
    }
    
    // ========================================
    // g2k: Map-agent cross-attention edges
    // Index: polygon_idx -> agent_idx (agent_major: a * T + t)
    // ========================================
    for (int a = 0; a < num_agents; a++) {
        for (int t = 0; t < num_intervals; t++) {
            int agent_idx = a * num_intervals + t;
            if (!agent_valid_mask[agent_idx]) continue;
            
            float ax = agent_positions[agent_idx * 2];
            float ay = agent_positions[agent_idx * 2 + 1];
            float ah = agent_headings[agent_idx];
            
            for (int p = 0; p < num_polygons; p++) {
                float px = polygon_positions[p * 2];
                float py = polygon_positions[p * 2 + 1];
                float ph = polygon_headings[p];
                
                float dist = std::sqrt((ax - px) * (ax - px) + (ay - py) * (ay - py));
                
                if (dist <= polygon_radius) {
                    // Transform polygon position to agent local coordinate
                    float dx = px - ax;
                    float dy = py - ay;
                    auto [local_x, local_y] = transformToLocal(dx, dy, ah);
                    float length = std::sqrt(local_x * local_x + local_y * local_y) + 1e-8f;
                    float theta = std::atan2(local_y, local_x);
                    float heading_diff = wrapAngle(ph - ah);
                    
                    g2k_edges_src.push_back(p);
                    g2k_edges_dst.push_back(agent_idx);
                    
                    // Edge attr: [length, cos(theta), sin(theta), cos(heading), sin(heading), heading_valid]
                    g2k_edge_attr.insert(g2k_edge_attr.end(), {
                        length, std::cos(theta), std::sin(theta),
                        std::cos(heading_diff), std::sin(heading_diff), 1.0f  // heading_valid
                    });
                }
            }
        }
    }
    
    // ========================================
    // k2k_a: Agent-agent attention edges
    // IMPORTANT: Index format is time_major -> idx = timestep * MAX_AGENTS + agent
    // This matches StepModel which transposes to [T, N, D] for k2k_a attention
    // ========================================
    for (int t = 0; t < num_intervals; t++) {
        for (int a1 = 0; a1 < num_agents; a1++) {
            int idx1_agent_major = a1 * num_intervals + t;
            if (!agent_valid_mask[idx1_agent_major]) continue;
            
            float a1x = agent_positions[idx1_agent_major * 2];
            float a1y = agent_positions[idx1_agent_major * 2 + 1];
            float a1h = agent_headings[idx1_agent_major];
            
            for (int a2 = 0; a2 < num_agents; a2++) {
                if (a1 == a2) continue;
                
                int idx2_agent_major = a2 * num_intervals + t;
                if (!agent_valid_mask[idx2_agent_major]) continue;
                
                float a2x = agent_positions[idx2_agent_major * 2];
                float a2y = agent_positions[idx2_agent_major * 2 + 1];
                float a2h = agent_headings[idx2_agent_major];
                
                float dist = std::sqrt((a1x - a2x) * (a1x - a2x) + (a1y - a2y) * (a1y - a2y));
                
                if (dist <= agent_radius) {
                    // time_major index for k2k_a: t * MAX_AGENTS + agent
                    int src_idx = t * MAX_AGENTS + a1;
                    int dst_idx = t * MAX_AGENTS + a2;
                    
                    // Transform a1 position to a2 local coordinate
                    float dx = a1x - a2x;
                    float dy = a1y - a2y;
                    auto [local_x, local_y] = transformToLocal(dx, dy, a2h);
                    float length = std::sqrt(local_x * local_x + local_y * local_y) + 1e-8f;
                    float theta = std::atan2(local_y, local_x);
                    float heading_diff = wrapAngle(a1h - a2h);
                    
                    k2k_a_edges_src.push_back(src_idx);
                    k2k_a_edges_dst.push_back(dst_idx);
                    
                    // Edge attr: [length, cos(theta), sin(theta), cos(heading), sin(heading)]
                    k2k_a_edge_attr.insert(k2k_a_edge_attr.end(), {
                        length, std::cos(theta), std::sin(theta),
                        std::cos(heading_diff), std::sin(heading_diff)
                    });
                }
            }
        }
    }
    
    // Reshape edge indices to [2, num_edges] format
    size_t n_k2k_t = k2k_t_edges_src.size();
    k2k_t_edge_index.resize(2 * n_k2k_t);
    for (size_t i = 0; i < n_k2k_t; i++) {
        k2k_t_edge_index[i] = k2k_t_edges_src[i];
        k2k_t_edge_index[n_k2k_t + i] = k2k_t_edges_dst[i];
    }
    
    size_t n_g2k = g2k_edges_src.size();
    g2k_edge_index.resize(2 * n_g2k);
    for (size_t i = 0; i < n_g2k; i++) {
        g2k_edge_index[i] = g2k_edges_src[i];
        g2k_edge_index[n_g2k + i] = g2k_edges_dst[i];
    }
    
    size_t n_k2k_a = k2k_a_edges_src.size();
    k2k_a_edge_index.resize(2 * n_k2k_a);
    for (size_t i = 0; i < n_k2k_a; i++) {
        k2k_a_edge_index[i] = k2k_a_edges_src[i];
        k2k_a_edge_index[n_k2k_a + i] = k2k_a_edges_dst[i];
    }
}

std::unordered_map<int, std::unordered_map<int, TokenEntry>> 
loadTokenDictionary(const std::string& path) {
    std::unordered_map<int, std::unordered_map<int, TokenEntry>> result;
    
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open token dictionary: " + path);
    }
    
    // Read number of tokens
    int32_t num_tokens;
    file.read(reinterpret_cast<char*>(&num_tokens), sizeof(int32_t));
    
    if (num_tokens != NUM_TOKENS) {
        throw std::runtime_error("Token count mismatch: expected " + 
                                 std::to_string(NUM_TOKENS) + ", got " + 
                                 std::to_string(num_tokens));
    }
    
    // Read tokens for each agent type: 0=Vehicle, 1=Pedestrian, 2=Bicycle
    for (int agent_type = 0; agent_type < 3; agent_type++) {
        for (int token_id = 0; token_id < num_tokens; token_id++) {
            float delta_x, delta_y, delta_heading;
            file.read(reinterpret_cast<char*>(&delta_x), sizeof(float));
            file.read(reinterpret_cast<char*>(&delta_y), sizeof(float));
            file.read(reinterpret_cast<char*>(&delta_heading), sizeof(float));
            
            result[agent_type][token_id] = {delta_x, delta_y, delta_heading};
        }
    }
    
    std::cout << "Loaded token dictionary: " << num_tokens << " tokens x 3 agent types" << std::endl;
    
    return result;
}

} // namespace planr1


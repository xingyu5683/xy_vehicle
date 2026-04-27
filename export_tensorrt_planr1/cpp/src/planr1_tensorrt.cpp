/*
 * Plan-R1 TensorRT Inference Implementation
 */

#include "planr1_tensorrt.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>

// For loading plugins
#include <dlfcn.h>

namespace planr1 {

// Helper functions to create Dims (TensorRT 10 compatible)
inline nvinfer1::Dims makeDims1(int d0) {
    nvinfer1::Dims dims;
    dims.nbDims = 1;
    dims.d[0] = d0;
    return dims;
}

inline nvinfer1::Dims makeDims2(int d0, int d1) {
    nvinfer1::Dims dims;
    dims.nbDims = 2;
    dims.d[0] = d0;
    dims.d[1] = d1;
    return dims;
}

inline nvinfer1::Dims makeDims3(int d0, int d1, int d2) {
    nvinfer1::Dims dims;
    dims.nbDims = 3;
    dims.d[0] = d0;
    dims.d[1] = d1;
    dims.d[2] = d2;
    return dims;
}

// ============================================================================
// TRTLogger Implementation
// ============================================================================

void TRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= mSeverity) {
        const char* severityStr = "";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: severityStr = "[F]"; break;
            case Severity::kERROR:          severityStr = "[E]"; break;
            case Severity::kWARNING:        severityStr = "[W]"; break;
            case Severity::kINFO:           severityStr = "[I]"; break;
            case Severity::kVERBOSE:        severityStr = "[V]"; break;
        }
        std::cout << "[TRT] " << severityStr << " " << msg << std::endl;
    }
}

// ============================================================================
// PlanR1TensorRT Implementation
// ============================================================================

PlanR1TensorRT::PlanR1TensorRT(
    const std::string& map_encoder_path,
    const std::string& step_model_path,
    const std::string& token_dict_path,
    const std::string& plugin_path
) {
    // Load custom plugin if provided
    if (!plugin_path.empty()) {
        void* handle = dlopen(plugin_path.c_str(), RTLD_LAZY);
        if (!handle) {
            throw std::runtime_error("Failed to load plugin: " + plugin_path + " - " + dlerror());
        }
        std::cout << "Loaded plugin: " << plugin_path << std::endl;
    }
    
    // Create CUDA stream
    cudaStreamCreate(&mStream);
    
    // Create runtime
    mRuntime.reset(nvinfer1::createInferRuntime(mLogger));
    if (!mRuntime) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }
    
    // Load engines
    std::cout << "Loading map encoder: " << map_encoder_path << std::endl;
    mMapEncoderEngine.reset(loadEngine(map_encoder_path));
    if (!mMapEncoderEngine) {
        throw std::runtime_error("Failed to load map encoder engine");
    }
    
    std::cout << "Loading step model: " << step_model_path << std::endl;
    mStepEngine.reset(loadEngine(step_model_path));
    if (!mStepEngine) {
        throw std::runtime_error("Failed to load step model engine");
    }
    
    // Create execution contexts
    mMapEncoderContext.reset(mMapEncoderEngine->createExecutionContext());
    mStepContext.reset(mStepEngine->createExecutionContext());
    
    if (!mMapEncoderContext || !mStepContext) {
        throw std::runtime_error("Failed to create execution contexts");
    }
    
    // Load token dictionary
    mTokenDict = loadTokenDict(token_dict_path);
    std::cout << "Loaded token dictionary: " << mTokenDict[0].size() << " tokens" << std::endl;
    
    std::cout << "PlanR1TensorRT initialized successfully" << std::endl;
}

PlanR1TensorRT::~PlanR1TensorRT() {
    freeBuffers();
    if (mStream) {
        cudaStreamDestroy(mStream);
    }
}

nvinfer1::ICudaEngine* PlanR1TensorRT::loadEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open engine file: " + path);
    }
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> data(size);
    file.read(data.data(), size);
    
    return mRuntime->deserializeCudaEngine(data.data(), size);
}

TokenDict PlanR1TensorRT::loadTokenDict(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open token dictionary: " + path);
    }
    
    int32_t num_tokens;
    file.read(reinterpret_cast<char*>(&num_tokens), sizeof(num_tokens));
    
    TokenDict dict(3);  // 3 agent types
    for (int type = 0; type < 3; ++type) {
        dict[type].resize(num_tokens);
        for (int i = 0; i < num_tokens; ++i) {
            file.read(reinterpret_cast<char*>(dict[type][i].data()), 3 * sizeof(float));
        }
    }
    
    return dict;
}

std::array<float, 3> PlanR1TensorRT::decodeToken(int64_t token, int64_t agent_type) {
    if (token < 0 || token >= static_cast<int64_t>(mTokenDict[0].size())) {
        return {0.0f, 0.0f, 0.0f};
    }
    int type_idx = std::clamp(static_cast<int>(agent_type), 0, 2);
    return mTokenDict[type_idx][token];
}

void PlanR1TensorRT::allocateBuffers() {
    // Pre-allocate common buffers
    // This is called lazily on first inference
}

void PlanR1TensorRT::freeBuffers() {
    for (auto& [name, ptr] : mDeviceBuffers) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    mDeviceBuffers.clear();
}

std::vector<float> PlanR1TensorRT::runMapEncoder(const MapEncoderInput& input) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Get binding indices
    int numBindings = mMapEncoderEngine->getNbIOTensors();
    std::vector<void*> bindings(numBindings);
    std::vector<void*> devicePtrs;
    
    // Helper to allocate and copy input
    auto setInput = [&](const char* name, const void* data, size_t bytes, nvinfer1::Dims dims) {
        void* dPtr;
        cudaMalloc(&dPtr, bytes);
        cudaMemcpyAsync(dPtr, data, bytes, cudaMemcpyHostToDevice, mStream);
        devicePtrs.push_back(dPtr);
        mMapEncoderContext->setTensorAddress(name, dPtr);
        mMapEncoderContext->setInputShape(name, dims);
    };
    
    // Set inputs
    setInput("polyline_position", input.polyline_position.data(), 
             MAX_POLYLINES * 2 * sizeof(float), makeDims2(MAX_POLYLINES, 2));
    setInput("polyline_heading", input.polyline_heading.data(),
             MAX_POLYLINES * sizeof(float), makeDims1(MAX_POLYLINES));
    setInput("polyline_length", input.polyline_length.data(),
             MAX_POLYLINES * sizeof(float), makeDims1(MAX_POLYLINES));
    setInput("polygon_position", input.polygon_position.data(),
             MAX_POLYGONS * 2 * sizeof(float), makeDims2(MAX_POLYGONS, 2));
    setInput("polygon_heading", input.polygon_heading.data(),
             MAX_POLYGONS * sizeof(float), makeDims1(MAX_POLYGONS));
    setInput("polygon_speed_limit", input.polygon_speed_limit.data(),
             MAX_POLYGONS * sizeof(float), makeDims1(MAX_POLYGONS));
    setInput("polygon_speed_limit_valid", input.polygon_speed_limit_valid.data(),
             MAX_POLYGONS * sizeof(float), makeDims1(MAX_POLYGONS));
    setInput("polygon_type", input.polygon_type.data(),
             MAX_POLYGONS * sizeof(int64_t), makeDims1(MAX_POLYGONS));
    setInput("polygon_traffic_light", input.polygon_traffic_light.data(),
             MAX_POLYGONS * sizeof(int64_t), makeDims1(MAX_POLYGONS));
    setInput("polygon_on_route", input.polygon_on_route.data(),
             MAX_POLYGONS * sizeof(int64_t), makeDims1(MAX_POLYGONS));
    setInput("l2g_edge_index", input.l2g_edge_index.data(),
             2 * MAX_POLYLINES * sizeof(int64_t), makeDims2(2, MAX_POLYLINES));
    setInput("left_edge_index", input.left_edge_index.data(),
             input.left_edge_index.size() * sizeof(int64_t), 
             makeDims2(2, static_cast<int>(input.left_edge_index.size() / 2)));
    setInput("right_edge_index", input.right_edge_index.data(),
             input.right_edge_index.size() * sizeof(int64_t),
             makeDims2(2, static_cast<int>(input.right_edge_index.size() / 2)));
    setInput("incoming_edge_index", input.incoming_edge_index.data(),
             input.incoming_edge_index.size() * sizeof(int64_t),
             makeDims2(2, static_cast<int>(input.incoming_edge_index.size() / 2)));
    setInput("outgoing_edge_index", input.outgoing_edge_index.data(),
             input.outgoing_edge_index.size() * sizeof(int64_t),
             makeDims2(2, static_cast<int>(input.outgoing_edge_index.size() / 2)));
    
    // Allocate output
    void* outputPtr;
    size_t outputBytes = MAX_POLYGONS * EMBED_DIM * sizeof(float);
    cudaMalloc(&outputPtr, outputBytes);
    devicePtrs.push_back(outputPtr);
    mMapEncoderContext->setTensorAddress("polygon_embs", outputPtr);
    
    // Execute
    bool success = mMapEncoderContext->enqueueV3(mStream);
    if (!success) {
        for (auto ptr : devicePtrs) cudaFree(ptr);
        throw std::runtime_error("Map encoder execution failed");
    }
    
    // Copy output back
    std::vector<float> output(MAX_POLYGONS * EMBED_DIM);
    cudaMemcpyAsync(output.data(), outputPtr, outputBytes, cudaMemcpyDeviceToHost, mStream);
    cudaStreamSynchronize(mStream);
    
    // Free temporary buffers
    for (auto ptr : devicePtrs) cudaFree(ptr);
    
    auto end = std::chrono::high_resolution_clock::now();
    mLastMapEncoderTime = std::chrono::duration<float, std::milli>(end - start).count();
    
    return output;
}

std::vector<float> PlanR1TensorRT::runSingleStep(const StepModelInput& input) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<void*> devicePtrs;
    
    // Helper to allocate and copy input
    auto setInput = [&](const char* name, const void* data, size_t bytes, nvinfer1::Dims dims) {
        void* dPtr;
        cudaMalloc(&dPtr, bytes);
        cudaMemcpyAsync(dPtr, data, bytes, cudaMemcpyHostToDevice, mStream);
        devicePtrs.push_back(dPtr);
        mStepContext->setTensorAddress(name, dPtr);
        mStepContext->setInputShape(name, dims);
    };
    
    int N = input.num_agents;
    int T = input.num_intervals;
    
    // Set inputs
    setInput("agent_token", input.agent_token.data(),
             N * T * sizeof(int64_t), makeDims2(N, T));
    setInput("agent_position", input.agent_position.data(),
             N * T * 2 * sizeof(float), makeDims3(N, T, 2));
    setInput("agent_heading", input.agent_heading.data(),
             N * T * sizeof(float), makeDims2(N, T));
    setInput("agent_velocity", input.agent_velocity.data(),
             N * T * 2 * sizeof(float), makeDims3(N, T, 2));
    setInput("agent_type", input.agent_type.data(),
             N * sizeof(int64_t), makeDims1(N));
    setInput("agent_valid_mask", input.agent_valid_mask.data(),
             N * T * sizeof(float), makeDims2(N, T));
    setInput("polygon_embs", input.polygon_embs.data(),
             MAX_POLYGONS * EMBED_DIM * sizeof(float), makeDims2(MAX_POLYGONS, EMBED_DIM));
    
    // Dynamic edge inputs
    int E1 = input.k2k_t_edge_index.size() / 2;
    int E2 = input.g2k_edge_index.size() / 2;
    int E3 = input.k2k_a_edge_index.size() / 2;
    
    if (E1 > 0) {
        setInput("k2k_t_edge_index", input.k2k_t_edge_index.data(),
                 2 * E1 * sizeof(int64_t), makeDims2(2, E1));
        setInput("k2k_t_edge_attr", input.k2k_t_edge_attr.data(),
                 E1 * 6 * sizeof(float), makeDims2(E1, 6));
    }
    
    if (E2 > 0) {
        setInput("g2k_edge_index", input.g2k_edge_index.data(),
                 2 * E2 * sizeof(int64_t), makeDims2(2, E2));
        setInput("g2k_edge_attr", input.g2k_edge_attr.data(),
                 E2 * 6 * sizeof(float), makeDims2(E2, 6));
    }
    
    if (E3 > 0) {
        setInput("k2k_a_edge_index", input.k2k_a_edge_index.data(),
                 2 * E3 * sizeof(int64_t), makeDims2(2, E3));
        setInput("k2k_a_edge_attr", input.k2k_a_edge_attr.data(),
                 E3 * 5 * sizeof(float), makeDims2(E3, 5));
    }
    
    // Allocate output
    void* outputPtr;
    size_t outputBytes = MAX_AGENTS * NUM_TOKENS * sizeof(float);
    cudaMalloc(&outputPtr, outputBytes);
    devicePtrs.push_back(outputPtr);
    mStepContext->setTensorAddress("logits", outputPtr);
    
    // Execute
    bool success = mStepContext->enqueueV3(mStream);
    if (!success) {
        for (auto ptr : devicePtrs) cudaFree(ptr);
        throw std::runtime_error("Step model execution failed");
    }
    
    // Copy output back
    std::vector<float> output(MAX_AGENTS * NUM_TOKENS);
    cudaMemcpyAsync(output.data(), outputPtr, outputBytes, cudaMemcpyDeviceToHost, mStream);
    cudaStreamSynchronize(mStream);
    
    // Free temporary buffers
    for (auto ptr : devicePtrs) cudaFree(ptr);
    
    auto end = std::chrono::high_resolution_clock::now();
    mLastStepTime = std::chrono::duration<float, std::milli>(end - start).count();
    
    return output;
}

InferenceOutput PlanR1TensorRT::runInference(StepModelInput& input) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    InferenceOutput output;
    output.trajectories.resize(MAX_AGENTS * NUM_FUTURE_STEPS * 3, 0.0f);
    output.predicted_tokens.resize(MAX_AGENTS * NUM_FUTURE_STEPS, 0);
    
    // Track positions and headings
    std::vector<float> positions(MAX_AGENTS * 2, 0.0f);
    std::vector<float> headings(MAX_AGENTS, 0.0f);
    
    float total_step_time = 0.0f;
    
    // Autoregressive loop
    for (int step = 0; step < NUM_FUTURE_STEPS; ++step) {
        // Run single step
        auto logits = runSingleStep(input);
        total_step_time += mLastStepTime;
        
        // Argmax to get predicted tokens
        for (int a = 0; a < MAX_AGENTS; ++a) {
            int best_token = 0;
            float best_score = logits[a * NUM_TOKENS];
            
            for (int t = 1; t < NUM_TOKENS; ++t) {
                float score = logits[a * NUM_TOKENS + t];
                if (score > best_score) {
                    best_score = score;
                    best_token = t;
                }
            }
            
            output.predicted_tokens[a * NUM_FUTURE_STEPS + step] = best_token;
            
            // Decode token
            auto delta = decodeToken(best_token, input.agent_type[a]);
            
            // Update position (in local frame, then rotate to global)
            float heading = headings[a];
            float cos_h = std::cos(heading);
            float sin_h = std::sin(heading);
            
            float dx_global = delta[0] * cos_h - delta[1] * sin_h;
            float dy_global = delta[0] * sin_h + delta[1] * cos_h;
            
            positions[a * 2] += dx_global;
            positions[a * 2 + 1] += dy_global;
            headings[a] += delta[2];
            
            // Store trajectory
            output.trajectories[(a * NUM_FUTURE_STEPS + step) * 3 + 0] = positions[a * 2];
            output.trajectories[(a * NUM_FUTURE_STEPS + step) * 3 + 1] = positions[a * 2 + 1];
            output.trajectories[(a * NUM_FUTURE_STEPS + step) * 3 + 2] = headings[a];
        }
        
        // Update input tokens for next step
        int T = input.num_intervals;
        for (int a = 0; a < MAX_AGENTS; ++a) {
            // Shift tokens left
            for (int t = 0; t < T - 1; ++t) {
                input.agent_token[a * T + t] = input.agent_token[a * T + t + 1];
            }
            // Add new token
            input.agent_token[a * T + T - 1] = output.predicted_tokens[a * NUM_FUTURE_STEPS + step];
        }
        
        // Update positions and headings in input
        for (int a = 0; a < MAX_AGENTS; ++a) {
            // Shift positions/headings left and add new
            for (int t = 0; t < T - 1; ++t) {
                input.agent_position[(a * T + t) * 2] = input.agent_position[(a * T + t + 1) * 2];
                input.agent_position[(a * T + t) * 2 + 1] = input.agent_position[(a * T + t + 1) * 2 + 1];
                input.agent_heading[a * T + t] = input.agent_heading[a * T + t + 1];
            }
            input.agent_position[(a * T + T - 1) * 2] = positions[a * 2];
            input.agent_position[(a * T + T - 1) * 2 + 1] = positions[a * 2 + 1];
            input.agent_heading[a * T + T - 1] = headings[a];
        }
        
        // TODO: Recompute edges based on new positions
        // For now, we use the same edges (simplified)
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    
    output.map_encoder_time_ms = mLastMapEncoderTime;
    output.step_time_ms = total_step_time;
    output.total_time_ms = std::chrono::duration<float, std::milli>(total_end - total_start).count();
    
    return output;
}

// ============================================================================
// Utility Functions
// ============================================================================

TokenDict loadTokenDictionary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open token dictionary: " + path);
    }
    
    int32_t num_tokens;
    file.read(reinterpret_cast<char*>(&num_tokens), sizeof(num_tokens));
    
    TokenDict dict(3);
    for (int type = 0; type < 3; ++type) {
        dict[type].resize(num_tokens);
        for (int i = 0; i < num_tokens; ++i) {
            file.read(reinterpret_cast<char*>(dict[type][i].data()), 3 * sizeof(float));
        }
    }
    
    return dict;
}

inline float wrapAngle(float angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

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
) {
    k2k_t_edge_index.clear();
    k2k_t_edge_attr.clear();
    g2k_edge_index.clear();
    g2k_edge_attr.clear();
    k2k_a_edge_index.clear();
    k2k_a_edge_attr.clear();
    
    std::vector<int64_t> k2k_t_src, k2k_t_dst;
    std::vector<int64_t> g2k_src, g2k_dst;
    std::vector<int64_t> k2k_a_src, k2k_a_dst;
    
    auto transformToLocal = [](float dx, float dy, float heading) -> std::pair<float, float> {
        float cos_h = std::cos(-heading);
        float sin_h = std::sin(-heading);
        return {dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h};
    };
    
    // k2k_t: Temporal self-attention
    for (int a = 0; a < num_agents; ++a) {
        for (int t_dst = 0; t_dst < num_intervals; ++t_dst) {
            int dst_idx = a * num_intervals + t_dst;
            if (!agent_valid_mask[dst_idx]) continue;
            
            float dst_x = agent_positions[dst_idx * 2];
            float dst_y = agent_positions[dst_idx * 2 + 1];
            float dst_h = agent_headings[dst_idx];
            
            for (int t_src = std::max(0, t_dst - 6); t_src <= t_dst; ++t_src) {
                int src_idx = a * num_intervals + t_src;
                if (!agent_valid_mask[src_idx]) continue;
                
                float src_x = agent_positions[src_idx * 2];
                float src_y = agent_positions[src_idx * 2 + 1];
                float src_h = agent_headings[src_idx];
                
                auto [local_x, local_y] = transformToLocal(src_x - dst_x, src_y - dst_y, dst_h);
                float length = std::sqrt(local_x * local_x + local_y * local_y) + 1e-8f;
                float theta = std::atan2(local_y, local_x);
                float heading_diff = wrapAngle(src_h - dst_h);
                float dt = static_cast<float>(t_src - t_dst) * interval_frames;
                
                k2k_t_src.push_back(src_idx);
                k2k_t_dst.push_back(dst_idx);
                k2k_t_edge_attr.insert(k2k_t_edge_attr.end(), {
                    length, std::cos(theta), std::sin(theta),
                    std::cos(heading_diff), std::sin(heading_diff), dt
                });
            }
        }
    }
    
    // g2k: Map-agent cross-attention
    for (int a = 0; a < num_agents; ++a) {
        for (int t = 0; t < num_intervals; ++t) {
            int agent_idx = a * num_intervals + t;
            if (!agent_valid_mask[agent_idx]) continue;
            
            float ax = agent_positions[agent_idx * 2];
            float ay = agent_positions[agent_idx * 2 + 1];
            float ah = agent_headings[agent_idx];
            
            for (int p = 0; p < num_polygons; ++p) {
                float px = polygon_positions[p * 2];
                float py = polygon_positions[p * 2 + 1];
                float ph = polygon_headings[p];
                
                float dist = std::sqrt((ax - px) * (ax - px) + (ay - py) * (ay - py));
                
                if (dist <= polygon_radius) {
                    auto [local_x, local_y] = transformToLocal(px - ax, py - ay, ah);
                    float length = std::sqrt(local_x * local_x + local_y * local_y) + 1e-8f;
                    float theta = std::atan2(local_y, local_x);
                    float heading_diff = wrapAngle(ph - ah);
                    
                    g2k_src.push_back(p);
                    g2k_dst.push_back(agent_idx);
                    g2k_edge_attr.insert(g2k_edge_attr.end(), {
                        length, std::cos(theta), std::sin(theta),
                        std::cos(heading_diff), std::sin(heading_diff), 1.0f
                    });
                }
            }
        }
    }
    
    // k2k_a: Agent-agent attention (time-major indexing)
    for (int t = 0; t < num_intervals; ++t) {
        for (int a1 = 0; a1 < num_agents; ++a1) {
            int idx1 = a1 * num_intervals + t;
            if (!agent_valid_mask[idx1]) continue;
            
            float a1x = agent_positions[idx1 * 2];
            float a1y = agent_positions[idx1 * 2 + 1];
            float a1h = agent_headings[idx1];
            
            for (int a2 = 0; a2 < num_agents; ++a2) {
                if (a1 == a2) continue;
                
                int idx2 = a2 * num_intervals + t;
                if (!agent_valid_mask[idx2]) continue;
                
                float a2x = agent_positions[idx2 * 2];
                float a2y = agent_positions[idx2 * 2 + 1];
                float a2h = agent_headings[idx2];
                
                float dist = std::sqrt((a1x - a2x) * (a1x - a2x) + (a1y - a2y) * (a1y - a2y));
                
                if (dist <= agent_radius) {
                    // Time-major index
                    int src_idx = t * MAX_AGENTS + a1;
                    int dst_idx = t * MAX_AGENTS + a2;
                    
                    auto [local_x, local_y] = transformToLocal(a1x - a2x, a1y - a2y, a2h);
                    float length = std::sqrt(local_x * local_x + local_y * local_y) + 1e-8f;
                    float theta = std::atan2(local_y, local_x);
                    float heading_diff = wrapAngle(a1h - a2h);
                    
                    k2k_a_src.push_back(src_idx);
                    k2k_a_dst.push_back(dst_idx);
                    k2k_a_edge_attr.insert(k2k_a_edge_attr.end(), {
                        length, std::cos(theta), std::sin(theta),
                        std::cos(heading_diff), std::sin(heading_diff)
                    });
                }
            }
        }
    }
    
    // Reshape to [2, E] format
    auto reshape = [](const std::vector<int64_t>& src, const std::vector<int64_t>& dst,
                      std::vector<int64_t>& output) {
        size_t E = src.size();
        output.resize(2 * E);
        std::copy(src.begin(), src.end(), output.begin());
        std::copy(dst.begin(), dst.end(), output.begin() + E);
    };
    
    reshape(k2k_t_src, k2k_t_dst, k2k_t_edge_index);
    reshape(g2k_src, g2k_dst, g2k_edge_index);
    reshape(k2k_a_src, k2k_a_dst, k2k_a_edge_index);
}

}  // namespace planr1

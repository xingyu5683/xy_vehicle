/**
 * Plan-R1 ONNX Inference - Main Entry Point
 * 
 * Example usage of the PlanR1Inference class.
 */

#include "planr1_inference.h"
#include <iostream>
#include <fstream>
#include <chrono>

using namespace planr1;

/**
 * Load token dictionary from binary file.
 * Format: int32(num_tokens) + 3 agent_types * 1024 tokens * 3 floats (dx, dy, dheading)
 */
std::unordered_map<int, TokenEntry> loadTokenDict(const std::string& path) {
    // Use the library function to load, then flatten for the simple API
    auto full_dict = planr1::loadTokenDictionary(path);
    
    // Return Vehicle tokens as the default (type 0)
    std::unordered_map<int, TokenEntry> result;
    for (const auto& [token_id, entry] : full_dict[0]) {
        result[token_id] = entry;
    }
    return result;
}

/**
 * Create dummy map encoder input for testing.
 */
MapEncoderInput createDummyMapInput() {
    MapEncoderInput input;
    
    // Polyline features
    input.polyline_position.resize(MAX_POLYLINES * 2, 0.0f);
    input.polyline_heading.resize(MAX_POLYLINES, 0.0f);
    input.polyline_length.resize(MAX_POLYLINES, 1.0f);
    
    // Polygon features
    input.polygon_position.resize(MAX_POLYGONS * 2, 0.0f);
    input.polygon_heading.resize(MAX_POLYGONS, 0.0f);
    input.polygon_heading_valid.resize(MAX_POLYGONS, 1);
    input.polygon_type.resize(MAX_POLYGONS, 0);
    input.polygon_traffic_light.resize(MAX_POLYGONS, 0);
    input.polygon_speed_limit.resize(MAX_POLYGONS, 13.4f);  // ~30 mph
    input.polygon_speed_limit_valid_float.resize(MAX_POLYGONS, 1.0f);  // FLOAT type
    input.polygon_on_route.resize(MAX_POLYGONS, 1);
    
    // Create some sample polygon positions (road centerline)
    for (int i = 0; i < 50; i++) {
        input.polygon_position[i * 2] = static_cast<float>(i) * 2.0f;  // x
        input.polygon_position[i * 2 + 1] = 0.0f;  // y
    }
    
    // Edge indices (all 2 * 80 for consistency)
    input.left_edge_index.resize(2 * 80, 0);
    input.right_edge_index.resize(2 * 80, 0);
    input.incoming_edge_index.resize(2 * 80, 0);
    input.outgoing_edge_index.resize(2 * 80, 0);
    input.polyline_to_polygon_edge_index.resize(2 * MAX_POLYLINES, 0);  // l2g_edge_index
    
    input.num_polylines = 100;
    input.num_polygons = 50;
    
    return input;
}

/**
 * Create dummy step model input for testing.
 */
StepModelInput createDummyStepInput(const std::vector<float>& polygon_embs) {
    StepModelInput input;
    
    int num_agents = 5;
    int num_intervals = 4;  // 4 historical intervals
    
    // Agent features - agent_token is [MAX_AGENTS, num_intervals]
    input.agent_token.resize(MAX_AGENTS * num_intervals, 0);
    input.agent_type.resize(MAX_AGENTS, 0);  // All vehicles
    input.agent_box.resize(MAX_AGENTS * 4, 0.0f);
    input.agent_identity.resize(MAX_AGENTS, 1);
    input.agent_identity[0] = 0;  // First agent is ego
    
    // Set box sizes (length, width for each corner)
    for (int a = 0; a < num_agents; a++) {
        input.agent_box[a * 4] = 2.5f;   // front
        input.agent_box[a * 4 + 1] = 2.5f;  // rear
        input.agent_box[a * 4 + 2] = 1.0f;  // left
        input.agent_box[a * 4 + 3] = 1.0f;  // right
    }
    
    // Initialize token history with some dummy values
    for (int a = 0; a < num_agents; a++) {
        for (int t = 0; t < num_intervals; t++) {
            input.agent_token[a * num_intervals + t] = t;  // Dummy tokens
        }
    }
    
    // Polygon embeddings from map encoder
    input.polygon_embs = polygon_embs;
    
    // Compute edges (simplified - using dummy positions)
    // agent_positions: [num_agents * num_intervals * 2]
    std::vector<float> agent_positions(num_agents * num_intervals * 2);
    std::vector<float> agent_headings(num_agents * num_intervals, 0.0f);
    std::vector<bool> agent_valid_mask(num_agents * num_intervals, true);
    std::vector<float> polygon_positions(MAX_POLYGONS * 2, 0.0f);
    std::vector<float> polygon_headings(MAX_POLYGONS, 0.0f);
    
    for (int a = 0; a < num_agents; a++) {
        for (int t = 0; t < num_intervals; t++) {
            int idx = a * num_intervals + t;
            agent_positions[idx * 2] = static_cast<float>(a * 10 + t);  // x
            agent_positions[idx * 2 + 1] = static_cast<float>(a * 3);   // y
            agent_headings[idx] = 0.0f;
        }
    }
    for (int i = 0; i < 50; i++) {
        polygon_positions[i * 2] = static_cast<float>(i) * 2.0f;
    }
    
    computeEdges(
        agent_positions,
        agent_headings,
        agent_valid_mask,
        polygon_positions,
        polygon_headings,
        num_agents,
        50,  // num_polygons
        num_intervals,
        60.0f,  // agent_radius
        30.0f,  // polygon_radius
        5,      // interval_frames
        input.k2k_t_edge_index,
        input.k2k_t_edge_attr,
        input.g2k_edge_index,
        input.g2k_edge_attr,
        input.k2k_a_edge_index,
        input.k2k_a_edge_attr
    );
    
    input.num_agents = num_agents;
    input.num_intervals = num_intervals;
    input.num_k2k_t_edges = input.k2k_t_edge_index.size() / 2;
    input.num_g2k_edges = input.g2k_edge_index.size() / 2;
    input.num_k2k_a_edges = input.k2k_a_edge_index.size() / 2;
    
    return input;
}

void printUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <map_encoder.onnx> <step.onnx> <tokens.bin> [--cpu]" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  map_encoder.onnx  Path to the map encoder ONNX model" << std::endl;
    std::cout << "  step.onnx         Path to the step model ONNX model" << std::endl;
    std::cout << "  tokens.bin        Path to the token dictionary (binary format)" << std::endl;
    std::cout << "  --cpu             Use CPU instead of GPU (optional)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string map_encoder_path = argv[1];
    std::string step_model_path = argv[2];
    std::string token_dict_path = argv[3];
    bool use_gpu = true;
    
    for (int i = 4; i < argc; i++) {
        if (std::string(argv[i]) == "--cpu") {
            use_gpu = false;
        }
    }
    
    std::cout << "============================================================" << std::endl;
    std::cout << "Plan-R1 ONNX Inference" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Map Encoder: " << map_encoder_path << std::endl;
    std::cout << "Step Model:  " << step_model_path << std::endl;
    std::cout << "Token Dict:  " << token_dict_path << std::endl;
    std::cout << "Device:      " << (use_gpu ? "GPU (CUDA)" : "CPU") << std::endl;
    std::cout << std::endl;
    
    try {
        // Load token dictionary from binary file
        auto token_dict = loadTokenDict(token_dict_path);
        std::cout << "Token dictionary: " << token_dict.size() << " tokens loaded" << std::endl;
        
        // Initialize inference engine
        PlanR1Inference inference(map_encoder_path, step_model_path, token_dict, use_gpu);
        
        // Create dummy inputs
        std::cout << "\nCreating dummy inputs..." << std::endl;
        auto map_input = createDummyMapInput();
        
        // Run map encoder
        std::cout << "\nRunning map encoder..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        auto polygon_embs = inference.runMapEncoder(map_input);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  ✓ Map encoder completed in " << duration.count() << " ms" << std::endl;
        std::cout << "  ✓ Output shape: [" << polygon_embs.size() / HIDDEN_DIM << ", " << HIDDEN_DIM << "]" << std::endl;
        
        // Create step input
        auto step_input = createDummyStepInput(polygon_embs);
        std::cout << "\nEdge counts:" << std::endl;
        std::cout << "  k2k_t edges: " << step_input.num_k2k_t_edges << std::endl;
        std::cout << "  g2k edges:   " << step_input.num_g2k_edges << std::endl;
        std::cout << "  k2k_a edges: " << step_input.num_k2k_a_edges << std::endl;
        
        // Run full inference
        std::cout << "\nRunning autoregressive inference (16 steps)..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        auto output = inference.runInference(step_input);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "  ✓ Inference completed in " << duration.count() << " ms" << std::endl;
        std::cout << "  ✓ Average per step: " << duration.count() / NUM_FUTURE_STEPS << " ms" << std::endl;
        
        // Print results
        std::cout << "\nPredicted trajectories:" << std::endl;
        for (int a = 0; a < std::min(output.num_agents, 3); a++) {
            std::cout << "  Agent " << a << " (ego=" << (a == 0 ? "yes" : "no") << "):" << std::endl;
            for (int t = 0; t < NUM_FUTURE_STEPS; t += 4) {  // Print every 4th step
                int idx = a * NUM_FUTURE_STEPS * 2 + t * 2;
                std::cout << "    t=" << t << ": pos=(" 
                          << output.positions[idx] << ", " 
                          << output.positions[idx + 1] << "), heading=" 
                          << output.headings[a * NUM_FUTURE_STEPS + t] << std::endl;
            }
        }
        
        std::cout << "\n✓ Inference completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

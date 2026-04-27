/*
 * Plan-R1 TensorRT Inference Demo
 * 
 * Usage:
 *   ./planr1_tensorrt_demo map_encoder.trt step.trt tokens.bin [--plugin libscatter_add.so]
 */

#include "planr1_tensorrt.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

using namespace planr1;

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <map_encoder.trt> <step.trt> <tokens.bin> [options]\n"
              << "\nOptions:\n"
              << "  --plugin <path>   Path to custom plugin library\n"
              << "  --benchmark       Run benchmark mode (multiple iterations)\n"
              << "  --help            Show this message\n"
              << std::endl;
}

MapEncoderInput createDummyMapEncoderInput() {
    MapEncoderInput input;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    
    // Polyline features
    input.polyline_position.resize(MAX_POLYLINES * 2);
    input.polyline_heading.resize(MAX_POLYLINES);
    input.polyline_length.resize(MAX_POLYLINES);
    for (int i = 0; i < MAX_POLYLINES; ++i) {
        input.polyline_position[i * 2] = normal(gen) * 100;
        input.polyline_position[i * 2 + 1] = normal(gen) * 100;
        input.polyline_heading[i] = normal(gen);
        input.polyline_length[i] = std::abs(normal(gen)) + 0.1f;
    }
    
    // Polygon features
    input.polygon_position.resize(MAX_POLYGONS * 2);
    input.polygon_heading.resize(MAX_POLYGONS);
    input.polygon_speed_limit.resize(MAX_POLYGONS);
    input.polygon_speed_limit_valid.resize(MAX_POLYGONS);
    input.polygon_type.resize(MAX_POLYGONS);
    input.polygon_traffic_light.resize(MAX_POLYGONS);
    input.polygon_on_route.resize(MAX_POLYGONS);
    
    for (int i = 0; i < MAX_POLYGONS; ++i) {
        input.polygon_position[i * 2] = normal(gen) * 50;
        input.polygon_position[i * 2 + 1] = normal(gen) * 50;
        input.polygon_heading[i] = normal(gen);
        input.polygon_speed_limit[i] = uniform(gen) * 20;
        input.polygon_speed_limit_valid[i] = 1.0f;
        input.polygon_type[i] = 0;
        input.polygon_traffic_light[i] = 4;
        input.polygon_on_route[i] = 0;
    }
    
    // Edge indices
    constexpr int MAX_EDGES = 80;
    
    input.l2g_edge_index.resize(2 * MAX_POLYLINES);
    for (int i = 0; i < MAX_POLYLINES; ++i) {
        input.l2g_edge_index[i] = i;
        input.l2g_edge_index[MAX_POLYLINES + i] = i % MAX_POLYGONS;
    }
    
    auto createEdges = [&](std::vector<int64_t>& edges, int n) {
        edges.resize(2 * n);
        for (int i = 0; i < n; ++i) {
            edges[i] = i;
            edges[n + i] = (i + 1) % MAX_POLYGONS;
        }
    };
    
    createEdges(input.left_edge_index, MAX_EDGES);
    createEdges(input.right_edge_index, MAX_EDGES);
    createEdges(input.incoming_edge_index, MAX_EDGES);
    createEdges(input.outgoing_edge_index, MAX_EDGES);
    
    return input;
}

StepModelInput createDummyStepInput(const std::vector<float>& polygon_embs) {
    StepModelInput input;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal(0.0f, 1.0f);
    
    constexpr int NUM_INTERVALS = 5;
    
    input.num_agents = MAX_AGENTS;
    input.num_intervals = NUM_INTERVALS;
    
    // Agent features
    input.agent_token.resize(MAX_AGENTS * NUM_INTERVALS, 0);
    input.agent_position.resize(MAX_AGENTS * NUM_INTERVALS * 2);
    input.agent_heading.resize(MAX_AGENTS * NUM_INTERVALS);
    input.agent_velocity.resize(MAX_AGENTS * NUM_INTERVALS * 2);
    input.agent_type.resize(MAX_AGENTS, 0);
    input.agent_valid_mask.resize(MAX_AGENTS * NUM_INTERVALS, 1.0f);
    
    for (int a = 0; a < MAX_AGENTS; ++a) {
        for (int t = 0; t < NUM_INTERVALS; ++t) {
            int idx = a * NUM_INTERVALS + t;
            input.agent_position[idx * 2] = normal(gen) * 10;
            input.agent_position[idx * 2 + 1] = normal(gen) * 10;
            input.agent_heading[idx] = normal(gen) * 0.5f;
            input.agent_velocity[idx * 2] = normal(gen) * 5;
            input.agent_velocity[idx * 2 + 1] = normal(gen) * 2;
        }
    }
    
    // Polygon embeddings
    input.polygon_embs = polygon_embs;
    
    // Compute edges
    std::vector<bool> valid_mask(MAX_AGENTS * NUM_INTERVALS, true);
    std::vector<float> polygon_positions(MAX_POLYGONS * 2);
    std::vector<float> polygon_headings(MAX_POLYGONS);
    
    for (int i = 0; i < MAX_POLYGONS; ++i) {
        polygon_positions[i * 2] = normal(gen) * 50;
        polygon_positions[i * 2 + 1] = normal(gen) * 50;
        polygon_headings[i] = normal(gen);
    }
    
    computeEdges(
        input.agent_position,
        input.agent_heading,
        valid_mask,
        polygon_positions,
        polygon_headings,
        MAX_AGENTS,
        MAX_POLYGONS,
        NUM_INTERVALS,
        50.0f,  // agent_radius
        100.0f, // polygon_radius
        5,      // interval_frames
        input.k2k_t_edge_index,
        input.k2k_t_edge_attr,
        input.g2k_edge_index,
        input.g2k_edge_attr,
        input.k2k_a_edge_index,
        input.k2k_a_edge_attr
    );
    
    return input;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string map_encoder_path = argv[1];
    std::string step_model_path = argv[2];
    std::string token_dict_path = argv[3];
    std::string plugin_path;
    bool benchmark = false;
    
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--plugin" && i + 1 < argc) {
            plugin_path = argv[++i];
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    std::cout << "\n============================================================\n";
    std::cout << "Plan-R1 TensorRT FP16 Inference\n";
    std::cout << "============================================================\n";
    std::cout << "Map Encoder: " << map_encoder_path << "\n";
    std::cout << "Step Model:  " << step_model_path << "\n";
    std::cout << "Token Dict:  " << token_dict_path << "\n";
    if (!plugin_path.empty()) {
        std::cout << "Plugin:      " << plugin_path << "\n";
    }
    std::cout << std::endl;
    
    try {
        // Initialize inference engine
        PlanR1TensorRT inference(
            map_encoder_path,
            step_model_path,
            token_dict_path,
            plugin_path
        );
        
        // Create dummy inputs
        std::cout << "\nCreating dummy inputs...\n";
        auto map_input = createDummyMapEncoderInput();
        
        // Run map encoder
        std::cout << "\nRunning map encoder...\n";
        auto polygon_embs = inference.runMapEncoder(map_input);
        std::cout << "  ✓ Map encoder: " << std::fixed << std::setprecision(2)
                  << inference.getLastMapEncoderTimeMs() << " ms\n";
        std::cout << "  ✓ Output shape: [" << MAX_POLYGONS << ", " << EMBED_DIM << "]\n";
        
        // Create step input
        auto step_input = createDummyStepInput(polygon_embs);
        
        std::cout << "\nEdge counts:\n";
        std::cout << "  k2k_t edges: " << step_input.k2k_t_edge_index.size() / 2 << "\n";
        std::cout << "  g2k edges:   " << step_input.g2k_edge_index.size() / 2 << "\n";
        std::cout << "  k2k_a edges: " << step_input.k2k_a_edge_index.size() / 2 << "\n";
        
        // Run inference
        std::cout << "\nRunning autoregressive inference (16 steps)...\n";
        auto output = inference.runInference(step_input);
        
        std::cout << "  ✓ Step inference: " << std::fixed << std::setprecision(2)
                  << output.step_time_ms << " ms (total)\n";
        std::cout << "  ✓ Average per step: " << output.step_time_ms / NUM_FUTURE_STEPS << " ms\n";
        std::cout << "  ✓ Total time: " << output.total_time_ms << " ms\n";
        
        // Calculate frequency
        float freq = 1000.0f / (inference.getLastMapEncoderTimeMs() + output.step_time_ms);
        std::cout << "\n  → Inference frequency: " << std::fixed << std::setprecision(1)
                  << freq << " Hz\n";
        
        // Print sample trajectories
        std::cout << "\nPredicted trajectories:\n";
        for (int a = 0; a < 3; ++a) {
            std::cout << "  Agent " << a << " (ego=" << (a == 0 ? "yes" : "no") << "):\n";
            for (int step : {0, 4, 8, 12, 15}) {
                int idx = (a * NUM_FUTURE_STEPS + step) * 3;
                std::cout << "    t=" << step << ": pos=("
                          << std::fixed << std::setprecision(3)
                          << output.trajectories[idx] << ", "
                          << output.trajectories[idx + 1] << "), heading="
                          << output.trajectories[idx + 2] << "\n";
            }
        }
        
        // Benchmark mode
        if (benchmark) {
            std::cout << "\n============================================================\n";
            std::cout << "Benchmark Mode (100 iterations)\n";
            std::cout << "============================================================\n";
            
            constexpr int NUM_ITERS = 100;
            float total_map_time = 0.0f;
            float total_step_time = 0.0f;
            
            for (int i = 0; i < NUM_ITERS; ++i) {
                auto me_input = createDummyMapEncoderInput();
                auto pe = inference.runMapEncoder(me_input);
                total_map_time += inference.getLastMapEncoderTimeMs();
                
                auto si = createDummyStepInput(pe);
                auto out = inference.runInference(si);
                total_step_time += out.step_time_ms;
                
                if ((i + 1) % 10 == 0) {
                    std::cout << "  Iteration " << (i + 1) << "/" << NUM_ITERS << "\n";
                }
            }
            
            std::cout << "\nBenchmark Results:\n";
            std::cout << "  Avg MapEncoder: " << std::fixed << std::setprecision(2)
                      << total_map_time / NUM_ITERS << " ms\n";
            std::cout << "  Avg Step×16:    " << total_step_time / NUM_ITERS << " ms\n";
            std::cout << "  Avg Total:      " << (total_map_time + total_step_time) / NUM_ITERS << " ms\n";
            std::cout << "  Avg Frequency:  " << 1000.0f * NUM_ITERS / (total_map_time + total_step_time) 
                      << " Hz\n";
        }
        
        std::cout << "\n✓ Inference completed successfully!\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

// Usage:
//
//  g++ run_inf.cc -o ff_infer \
//      -I/path/to/onnxruntime/include \
//      -L/path/to/onnxruntime/lib -lonnxruntime
//
//  ./ff_infer path/to/model.onnx [path/to/model.json]

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <optional>
#include <cmath>

#include <nlohmann/json.hpp>

namespace ff_interface {

// Default overridden by JSON
static const std::vector<std::string> kDefaultFeatureOrder = {
    "pt",
    "eta",
    "mass",
    "seedingJet_pt",
    "seedingJet_eta",
    "seedingJet_mass",
    "decayMode_0",
    "decayMode_1",
    "decayMode_2",
    "decayMode_10",
    "decayMode_11",
    "btagPNetB",
    "btagPNetCvB",
    "btagPNetCvL",
    "btagPNetCvNotB",
    "btagPNetQvG"
};

static const std::vector<std::string> kDefaultContFeatureNames = {
    "pt",
    "eta",
    "mass",
    "seedingJet_pt",
    "seedingJet_eta",
    "seedingJet_mass",
    "btagPNetB",
    "btagPNetCvB",
    "btagPNetCvL",
    "btagPNetCvNotB",
    "btagPNetQvG"
};

class FFNetONNXRunner {
public:
    FFNetONNXRunner(const std::string& model_path,
                    const std::string& config_path = std::string())
        : env(ORT_LOGGING_LEVEL_WARNING, "FFNetInference"),
          session(nullptr),
          feature_order(kDefaultFeatureOrder),
          cont_feature_names(kDefaultContFeatureNames)
    {
        check_model_exists_(model_path);
        load_config_(model_path, config_path);
        init_feature_layout_();

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session = Ort::Session(env, model_path.c_str(), session_options);

        std::cout << "[info] ONNX model loaded from " << model_path << "\n";
        std::cout << "[info] feature_order size = " << feature_order.size() << "\n";
    }

    float compute_w_ff(const std::vector<float>& cont_features, int decayMode) const {
        if (cont_features.size() != cont_feature_names.size()) {
            std::cerr << "[warn] cont_features.size() = " << cont_features.size()
                      << " (expected " << cont_feature_names.size() << ")\n";
        }

        std::vector<float> cf = cont_features;
        if (cf.size() < cont_feature_names.size()) {
            cf.resize(cont_feature_names.size(), 0.0f);
        }

        auto raw_input = build_input_row_(cf, decayMode);
        return run_onnx_(raw_input);
    }


    float compute_w_ff(const std::map<std::string, float>& features,
                       std::optional<int> decayMode = std::nullopt) const
    {
        int dm = 0;
        if (decayMode.has_value()) {
            dm = decayMode.value();
        } else {
            auto it = features.find("decayMode");
            if (it == features.end()) {
                throw std::runtime_error(
                    "Mapping input must contain key 'decayMode' if no decayMode is passed explicitly.");
            }
            dm = static_cast<int>(std::lround(it->second));
        }

        std::vector<float> cf(cont_feature_names.size(), 0.0f);
        for (std::size_t i = 0; i < cont_feature_names.size(); ++i) {
            const auto& name = cont_feature_names[i];
            auto it = features.find(name);
            if (it != features.end()) {
                cf[i] = it->second;
            }
        }

        auto raw_input = build_input_row_(cf, dm);
        return run_onnx_(raw_input);
    }

private:
    Ort::Env env;
    mutable Ort::Session session;

    std::vector<std::string> feature_order;
    std::vector<std::string> cont_feature_names;

    std::unordered_map<std::string, std::size_t> feature_index;
    std::vector<std::pair<int, std::size_t>> decay_mode_indices; 

    static void check_model_exists_(const std::string& model_path) {
        std::ifstream f(model_path.c_str());
        if (!f.good()) {
            throw std::runtime_error("Could not find ONNX model file: " + model_path);
        }
    }
    static std::string default_config_path_(const std::string& model_path) {
        std::string cfg = model_path;
        const std::string onnx = ".onnx";
        if (cfg.size() >= onnx.size() &&
            cfg.compare(cfg.size() - onnx.size(), onnx.size(), onnx) == 0) {
            cfg.replace(cfg.size() - onnx.size(), onnx.size(), ".json");
        } else {
            cfg += ".json";
        }
        return cfg;
    }

    void load_config_(const std::string& model_path,
                      const std::string& config_path)
    {
        std::string cfg = config_path;
        if (cfg.empty()) {
            cfg = default_config_path_(model_path);
        }

        std::ifstream in(cfg.c_str());
        if (!in.good()) {
            std::cout << "[info] No config at " << cfg
                      << ", using built-in feature_order.\n";
            return;
        }

        std::cout << "[info] Loading input config from " << cfg << "\n";
        nlohmann::json j;
        in >> j;

        if (j.contains("feature_order")) {
            feature_order.clear();
            for (const auto& x : j["feature_order"]) {
                feature_order.push_back(x.get<std::string>());
            }
        }
        if (j.contains("cont_feature_names")) {
            cont_feature_names.clear();
            for (const auto& x : j["cont_feature_names"]) {
                cont_feature_names.push_back(x.get<std::string>());
            }
        }

        std::cout << "[info] Config loaded: feature_order=" << feature_order.size()
                  << ", cont_feature_names=" << cont_feature_names.size() << "\n";
    }

    void init_feature_layout_() {
        feature_index.clear();
        for (std::size_t i = 0; i < feature_order.size(); ++i) {
            feature_index[feature_order[i]] = i;
        }

        decay_mode_indices.clear();
        for (std::size_t i = 0; i < feature_order.size(); ++i) {
            const std::string& name = feature_order[i];
            if (name.rfind("decayMode_", 0) == 0) {
                const std::string suffix = name.substr(std::string("decayMode_").size());
                try {
                    int dm_val = std::stoi(suffix);
                    decay_mode_indices.emplace_back(dm_val, i);
                } catch (const std::exception&) {
                    std::cerr << "[warn] Could not parse decayMode from '" << name << "'\n";
                }
            }
        }

        if (decay_mode_indices.empty()) {
            std::cerr << "[warn] No decayMode_* features found in feature_order\n";
        }
    }

    std::vector<float> build_input_row_(const std::vector<float>& cont_vec,
                                        int decayMode) const
    {
        std::vector<float> raw_input(feature_order.size(), 0.0f);
        for (std::size_t i = 0; i < cont_feature_names.size() && i < cont_vec.size(); ++i) {
            const auto& name = cont_feature_names[i];
            auto it = feature_index.find(name);
            if (it != feature_index.end()) {
                raw_input[it->second] = cont_vec[i];
            }
        }
        if (decay_mode_indices.empty()) {
            std::cerr << "[warn] No decayMode_* entries; decayMode ignored\n";
        } else {
            bool matched = false;
            for (const auto& [dm_val, idx] : decay_mode_indices) {
                if (idx >= raw_input.size()) continue;
                if (dm_val == decayMode) {
                    raw_input[idx] = 1.0f;
                    matched = true;
                } else {
                    raw_input[idx] = 0.0f;
                }
            }
            if (!matched) {
                std::cerr << "[warn] decayMode " << decayMode
                          << " not in configured decayMode_*; one-hot will be all zeros\n";
            }
        }

        return raw_input;
    }

    float run_onnx_(const std::vector<float>& raw_input) const {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(raw_input.size())};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(raw_input.data()),
            raw_input.size(),
            input_shape.data(),
            input_shape.size()
        );

        const char* input_names[]  = {"raw_input"};
        const char* output_names[] = {"w_ff"};

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,  &input_tensor, 1,
            output_names, 1
        );

        float result = output_tensors[0].GetTensorMutableData<float>()[0];
        return result;
    }
};

std::unique_ptr<FFNetONNXRunner> g_ff_runner_instance;

void initialize_ff_runner(const std::string& model_path,
                          const std::string& config_path = std::string()) {
    g_ff_runner_instance = std::make_unique<FFNetONNXRunner>(model_path, config_path);
}

FFNetONNXRunner& get_ff_runner() {
    return *g_ff_runner_instance;
}

} // namespace ff_interface

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " model.onnx [model.json]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string config_path = (argc == 3) ? argv[2] : std::string();

    try {
        ff_interface::initialize_ff_runner(model_path, config_path);
        auto& runner = ff_interface::get_ff_runner();

        // Example 1: mapping with names 
        std::map<std::string, float> features_map = {
            {"pt",            45.0f},
            {"eta",           0.3f},
            {"mass",          1.2f},
            {"seedingJet_pt", 50.0f},
            {"seedingJet_eta",0.1f},
            {"seedingJet_mass",10.0f},
            {"btagPNetB",     0.2f},
            {"btagPNetCvB",   0.1f},
            {"btagPNetCvL",   0.4f},
            {"btagPNetCvNotB",0.3f},
            {"btagPNetQvG",   0.5f},
            {"decayMode",     1.0f}
        };

        float w_map = runner.compute_w_ff(features_map);

        // Example 2: plain vector 
        std::vector<float> cont_features = {
            45.0f,  // pt
            0.3f,   // eta
            1.2f,   // mass
            50.0f,  // seedingJet_pt
            0.1f,   // seedingJet_eta
            10.0f,  // seedingJet_mass
            0.2f,   // btagPNetB
            0.1f,   // btagPNetCvB
            0.4f,   // btagPNetCvL
            0.3f,   // btagPNetCvNotB
            0.5f    // btagPNetQvG
        };
        int decayMode = 1;

        float w_vec = runner.compute_w_ff(cont_features, decayMode);

        std::cout << "\n=== SUMMARY ===\n";
        std::cout << "Model path:      " << model_path << "\n";
        if (!config_path.empty())
            std::cout << "Config path:     " << config_path << "\n";
        std::cout << "w_ff (mapping):  " << w_map << "\n";
        std::cout << "w_ff (vector):   " << w_vec << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[error] " << ex.what() << "\n";
        return 1;
    }
}
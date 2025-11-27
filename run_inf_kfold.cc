// K-fold FF ONNX inference
//
// Usage:
//   g++ run_inf_kfold.cc -o ff_infer \
//       -I/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/include/onnxruntime \
//       -L/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/lib64 \
//       -lonnxruntime -std=c++17 -O2
//
//   ./ff_infer models
//
// In analysis (RDataFrame), call:
//
//   KFoldFFONNX kff("models");
//   auto w = kff.compute_w_ff_event(event, decayMode,
//                                   pt, eta, mass,
//                                   seedingJet_pt, seedingJet_eta, seedingJet_mass,
//                                   btagPNetB, btagPNetCvB, btagPNetCvL,
//                                   btagPNetCvNotB, btagPNetQvG);

#include "onnxruntime_cxx_api.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;

struct FoldConfig {
    std::vector<std::string> feature_order;
    std::map<std::string, std::size_t> feature_index;
    std::vector<int> decay_modes;
    std::vector<std::size_t> decay_indices;

    explicit FoldConfig(const std::vector<std::string>& fo)
        : feature_order(fo)
    {
        for (std::size_t i = 0; i < feature_order.size(); ++i) {
            const auto& name = feature_order[i];
            feature_index[name] = i;
            if (name.rfind("decayMode_", 0) == 0) {
                int dm = std::stoi(name.substr(std::string("decayMode_").size()));
                decay_modes.push_back(dm);
                decay_indices.push_back(i);
            }
        }
    }
};

class KFoldFFONNX {
public:
    explicit KFoldFFONNX(const std::string& model_dir)
        : env_(ORT_LOGGING_LEVEL_WARNING, "FF_KFold")
    {
        // folds from feature_order_fold*.json 
        std::vector<int> fold_ids;
        for (const auto& entry : fs::directory_iterator(model_dir)) {
            if (!entry.is_regular_file()) continue;
            auto name = entry.path().filename().string(); 
            const std::string prefix = "feature_order_fold";
            const std::string suffix = ".json";
            if (name.rfind(prefix, 0) == 0 &&
                name.size() > prefix.size() + suffix.size() &&
                name.substr(name.size() - suffix.size()) == suffix)
            {
                std::string num = name.substr(
                    prefix.size(),
                    name.size() - prefix.size() - suffix.size()
                );
                try {
                    int idx = std::stoi(num);
                    fold_ids.push_back(idx);
                    std::cout << "[info] Found fold ID " << idx << " from " << name << "\n";
                } catch (...) {
                }
            }
        }

        if (fold_ids.empty()) {
            throw std::runtime_error("No feature_order_fold*.json found in " + model_dir);
        }

        int max_fold = *std::max_element(fold_ids.begin(), fold_ids.end());
        n_folds_ = max_fold + 1;

        std::cout << "[info] Initialising KFoldFFONNX from " << model_dir
                  << " (n_folds=" << n_folds_ << ")\n";

        for (int f = 0; f < n_folds_; ++f) {
            std::string cfg_path  = model_dir + "/feature_order_fold" + std::to_string(f) + ".json";
            std::string onnx_path = model_dir + "/model_fold"        + std::to_string(f) + ".onnx";

            std::ifstream cf(cfg_path);
            if (!cf) {
                throw std::runtime_error("Cannot open feature-order JSON: " + cfg_path);
            }
            json j;
            cf >> j;
            std::vector<std::string> feature_order =
                j.at("feature_order").get<std::vector<std::string>>();

            fold_cfgs_.emplace_back(feature_order);

            std::ifstream mf(onnx_path);
            if (!mf) {
                throw std::runtime_error("Cannot open ONNX file: " + onnx_path);
            }

            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(1);
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            sessions_.emplace_back(
                std::make_unique<Ort::Session>(env_, onnx_path.c_str(), opts)
            );

            std::cout << "[info] Fold " << f << ": ONNX initialised from " << onnx_path << "\n";
        }

        const auto& fo0 = fold_cfgs_.front().feature_order;
        std::cout << "[info] Feature order (fold 0): ";
        for (std::size_t i = 0; i < fo0.size(); ++i) {
            std::cout << fo0[i];
            if (i + 1 < fo0.size()) std::cout << ", ";
        }
        std::cout << "\n";
    }

    int n_folds() const { return n_folds_; }

    float compute_w_ff_event(
        long long event_id,
        int decayMode,
        float pt,
        float eta,
        float mass,
        float seedingJet_pt,
        float seedingJet_eta,
        float seedingJet_mass,
        float btagPNetB,
        float btagPNetCvB,
        float btagPNetCvL,
        float btagPNetCvNotB,
        float btagPNetQvG)
    {
        int fold = static_cast<int>(event_id % n_folds_);
        if (fold < 0) fold += n_folds_;

        const auto& cfg = fold_cfgs_[fold];
        auto& sess      = *sessions_[fold];

        std::size_t n_feat = cfg.feature_order.size();
        std::vector<float> input_data(n_feat, 0.f);

        const char* names[] = {
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
        float vals[] = {
            pt,
            eta,
            mass,
            seedingJet_pt,
            seedingJet_eta,
            seedingJet_mass,
            btagPNetB,
            btagPNetCvB,
            btagPNetCvL,
            btagPNetCvNotB,
            btagPNetQvG
        };
        constexpr std::size_t n_scalar = sizeof(vals) / sizeof(vals[0]);

        for (std::size_t i = 0; i < n_scalar; ++i) {
            auto it = cfg.feature_index.find(names[i]);
            if (it == cfg.feature_index.end()) continue;
            input_data[it->second] = vals[i];
        }

        // decayMode one hot 
        for (std::size_t i = 0; i < cfg.decay_modes.size(); ++i) {
            int dm_val        = cfg.decay_modes[i];
            std::size_t col_j = cfg.decay_indices[i];
            input_data[col_j] = (decayMode == dm_val) ? 1.f : 0.f;
        }

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> shape{1, static_cast<int64_t>(n_feat)};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            input_data.data(),
            input_data.size(),
            shape.data(),
            shape.size()
        );

        const char* input_names[]  = {"raw_input"};
        const char* output_names[] = {"w_ff"};

        auto output_tensors = sess.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1
        );

        float* out_data = output_tensors[0].GetTensorMutableData<float>();
        return out_data[0];
    }

private:
    Ort::Env env_;
    int n_folds_{0};
    std::vector<std::unique_ptr<Ort::Session>> sessions_;
    std::vector<FoldConfig> fold_cfgs_;
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIR\n";
        return 1;
    }

    std::string model_dir = argv[1];

    try {
        KFoldFFONNX kff(model_dir);

        // example
        std::vector<long long> event_id = {1000, 1001, 1002, 1003, 1004};
        std::vector<int>       decay    = {0,    1,    2,    10,   11};

        std::cout << "event_id  fold  decayMode  w_ff\n";
        for (std::size_t i = 0; i < event_id.size(); ++i) {
            float w = kff.compute_w_ff_event(
                event_id[i],
                decay[i],
                45.f, 0.3f, 1.2f,
                50.f, 0.1f, 10.f,
                0.2f, 0.1f, 0.4f, 0.3f, 0.5f
            );
            int fold = static_cast<int>(event_id[i] % kff.n_folds());
            std::cout << event_id[i] << "  "
                      << fold << "  "
                      << decay[i] << "  "
                      << w << "\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[error] " << ex.what() << "\n";
        return 1;
    }
}
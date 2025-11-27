// K-fold FF ONNX inference
//
// Usage:
//   g++ run_inf_kfold.cc -o ff_infer \
//       -I/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/include/onnxruntime \
//       -L/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/lib64 \
//       -lonnxruntime -std=c++17 -O2
//
//   ./ff_infer models 5

#include "onnxruntime_cxx_api.h"
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

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
    KFoldFFONNX(const std::string& model_dir, int n_folds)
        : env_(ORT_LOGGING_LEVEL_WARNING, "FF_KFold"),
          n_folds_(n_folds)
    {
        if (n_folds_ <= 0) {
            throw std::runtime_error("n_folds must be > 0");
        }

        std::cout << "[info] Initialising KFoldFFONNX from " << model_dir
                  << " with n_folds=" << n_folds_ << "\n";

        for (int f = 0; f < n_folds_; ++f) {
            std::string onnx_path = model_dir + "/model_fold" + std::to_string(f) + ".onnx";
            std::string cfg_path  = model_dir + "/feature_order_fold" + std::to_string(f) + ".json";

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
        for (const auto& name : fo0) {
            if (name.rfind("decayMode_", 0) != 0) {
                scalar_features_.push_back(name);
            }
        }

        std::cout << "[info] Feature order (fold 0): ";
        for (std::size_t i = 0; i < fo0.size(); ++i) {
            std::cout << fo0[i];
            if (i + 1 < fo0.size()) std::cout << ", ";
        }
        std::cout << "\n";
    }

    std::vector<float> compute_w_ff(
        const std::vector<long long>& event_id,
        const std::map<std::string, std::vector<float>>& features)
    {
        std::size_t n = event_id.size();
        if (n == 0) {
            return {};
        }

        std::cout << "[info] compute_w_ff: n_events=" << n << "\n";

        auto dm_it = features.find("decayMode");
        if (dm_it == features.end()) {
            throw std::runtime_error("Missing feature 'decayMode'.");
        }
        if (dm_it->second.size() != n) {
            throw std::runtime_error("decayMode feature size mismatch.");
        }

        std::vector<int> decay_mode(n);
        for (std::size_t i = 0; i < n; ++i) {
            decay_mode[i] = static_cast<int>(std::lround(dm_it->second[i]));
        }

        for (const auto& name : scalar_features_) {
            auto it = features.find(name);
            if (it == features.end()) {
                throw std::runtime_error("Missing scalar feature: " + name);
            }
            if (it->second.size() != n) {
                throw std::runtime_error("Size mismatch for feature: " + name);
            }
        }

        std::vector<float> out(n, 0.f);

        for (int f = 0; f < n_folds_; ++f) {
            std::vector<std::size_t> idxs;
            idxs.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                if (static_cast<int>(event_id[i] % n_folds_) == f) {
                    idxs.push_back(i);
                }
            }
            if (idxs.empty()) continue;

            const auto& cfg  = fold_cfgs_[f];
            auto& sess       = *sessions_[f];

            std::size_t m      = idxs.size();
            std::size_t n_feat = cfg.feature_order.size();
            std::vector<float> input_data(m * n_feat, 0.f);

            for (const auto& name : scalar_features_) {
                const auto& col = features.at(name);
                auto fi_it = cfg.feature_index.find(name);
                if (fi_it == cfg.feature_index.end()) continue;
                std::size_t j = fi_it->second;
                for (std::size_t k = 0; k < m; ++k) {
                    std::size_t i = idxs[k];
                    input_data[k * n_feat + j] = col[i];
                }
            }

            // decayMode one-hot
            for (std::size_t dm_i = 0; dm_i < cfg.decay_modes.size(); ++dm_i) {
                int dm_val        = cfg.decay_modes[dm_i];
                std::size_t col_j = cfg.decay_indices[dm_i];
                for (std::size_t k = 0; k < m; ++k) {
                    std::size_t i = idxs[k];
                    input_data[k * n_feat + col_j] =
                        (decay_mode[i] == dm_val) ? 1.f : 0.f;
                }
            }

            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> shape{
                static_cast<int64_t>(m),
                static_cast<int64_t>(n_feat)
            };

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
            for (std::size_t k = 0; k < m; ++k) {
                out[idxs[k]] = out_data[k];
            }
        }

        return out;
    }

private:
    Ort::Env env_;
    int n_folds_;
    std::vector<std::unique_ptr<Ort::Session>> sessions_;
    std::vector<FoldConfig> fold_cfgs_;
    std::vector<std::string> scalar_features_;
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIR N_FOLDS\n";
        return 1;
    }

    std::string model_dir = argv[1];
    int n_folds = std::stoi(argv[2]);

    try {
        KFoldFFONNX kff(model_dir, n_folds);

        // example
        std::size_t n = 8;
        std::vector<long long> event_id{1000, 1000, 1002, 1003, 1004, 1005, 1006, 1006};

        auto fill = [n](float v) {
            return std::vector<float>(n, v);
        };

        std::map<std::string, std::vector<float>> feats;
        feats["decayMode"]      = {0.f, 1.f, 2.f, 10.f, 11.f, 0.f, 1.f, 2.f};
        feats["pt"]             = fill(45.0f);
        feats["eta"]            = fill(0.3f);
        feats["mass"]           = fill(1.2f);
        feats["seedingJet_pt"]  = fill(50.0f);
        feats["seedingJet_eta"] = fill(0.1f);
        feats["seedingJet_mass"]= fill(10.0f);
        feats["btagPNetB"]      = fill(0.2f);
        feats["btagPNetCvB"]    = fill(0.1f);
        feats["btagPNetCvL"]    = fill(0.4f);
        feats["btagPNetCvNotB"] = fill(0.3f);
        feats["btagPNetQvG"]    = fill(0.5f);

        auto w = kff.compute_w_ff(event_id, feats);

        std::cout << "event_id  fold  decayMode  w_ff\n";
        for (std::size_t i = 0; i < n; ++i) {
            int fold = static_cast<int>(event_id[i] % n_folds);
            int dm   = static_cast<int>(std::lround(feats["decayMode"][i]));
            std::cout << event_id[i] << "  "
                      << fold << "  "
                      << dm << "  "
                      << w[i] << "\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[error] " << ex.what() << "\n";
        return 1;
    }
}
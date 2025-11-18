// Usage: 

// g++ run_inf.cc -o ff_infer   -I/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/include   /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/lib64/libonnxruntime.so
// ./ff_infer models/Run3_2022EE/model.onnx

#include "/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/include/onnxruntime/onnxruntime_cxx_api.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <algorithm>

namespace ff_interface {

class FFNetONNXRunner {
public:
    FFNetONNXRunner(const std::string& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "FFNetInference"),
          session(nullptr),
          decay_modes_to_encode{0, 1, 2, 10, 11}
    {
        std::ifstream f(model_path.c_str());
        if (!f.good()) {
            throw std::runtime_error("Could not find ONNX model file: " + model_path);
        }
        f.close();

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session = Ort::Session(env, model_path.c_str(), session_options);
        std::cout << " ONNX model loaded from " << model_path << std::endl;

        std::cout << " Hard coded feature order (N=16):\n";
        for (size_t i = 0; i < feature_order.size(); ++i) {
            std::cout << "  [" << i << "] " << feature_order[i] << "\n";
        }
    }

    float compute_w_ff(const std::vector<float>& cont_features, int decayMode) const {
        if (cont_features.size() != 11) {
            std::cerr << " compute_w_ff: cont_features.size() = "
                      << cont_features.size()
                      << " (expected 11: pt,eta,mass,jet_pt,jet_eta,jet_mass,"
                      << "B,CvB,CvL,CvNotB,QvG)" << std::endl;
        }

        std::vector<float> raw_input(feature_order.size(), 0.0f);

        // kinematics
        if (cont_features.size() >= 6) {
            raw_input[0] = cont_features[0]; // pt
            raw_input[1] = cont_features[1]; // eta
            raw_input[2] = cont_features[2]; // mass
            raw_input[3] = cont_features[3]; // seedingJet_pt
            raw_input[4] = cont_features[4]; // seedingJet_eta
            raw_input[5] = cont_features[5]; // seedingJet_mass
        }

        // decayMode_* one hot
        encode_decay_mode_one_hot(decayMode, raw_input);

        // PNet
        if (cont_features.size() >= 11) {
            raw_input[11] = cont_features[6];  // btagPNetB
            raw_input[12] = cont_features[7];  // btagPNetCvB
            raw_input[13] = cont_features[8];  // btagPNetCvL
            raw_input[14] = cont_features[9];  // btagPNetCvNotB
            raw_input[15] = cont_features[10]; // btagPNetQvG
        }

        std::cout << " Built raw_input vector (size = " << raw_input.size() << ")\n";
        std::cout << " Features (name -> value):\n";
        for (size_t i = 0; i < raw_input.size(); ++i) {
            std::cout << "  " << feature_order[i] << " = " << raw_input[i] << "\n";
        }

        // ONNX input tensor
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

        std::cout << " Running ONNX inference...\n";
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,  &input_tensor, 1,
            output_names, 1
        );

        float result = output_tensors[0].GetTensorMutableData<float>()[0];
        std::cout << " w_ff = " << result << std::endl;
        return result;
    }

private:
    // ONNX feature order
    const std::vector<std::string> feature_order = {
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

    void encode_decay_mode_one_hot(int decayMode, std::vector<float>& raw_input) const {
        bool matched = false;
        std::string matched_name;

        for (size_t i = 0; i < decay_modes_to_encode.size(); ++i) {
            int dm = decay_modes_to_encode[i];
            size_t idx = 6 + i; // 6 to 10 in feature_order

            if (idx >= raw_input.size()) continue;

            if (dm == decayMode) {
                raw_input[idx] = 1.0f;
                matched = true;
                matched_name = feature_order[idx];
            } else {
                raw_input[idx] = 0.0f;
            }

            std::cout << " decayMode one-hot: " << feature_order[idx]
                      << " (dm=" << dm << ") -> " << raw_input[idx] << "\n";
        }

        if (!matched) {
            std::cout << " decayMode " << decayMode
                      << " not found in {0,1,2,10,11}; all decayMode_* = 0\n";
        } else {
            std::cout << " Encoded decayMode=" << decayMode
                      << " on " << matched_name << "\n";
        }
    }

private:
    Ort::Env env;
    mutable Ort::Session session;
    const std::vector<int> decay_modes_to_encode;
};

std::unique_ptr<FFNetONNXRunner> g_ff_runner_instance;

void initialize_ff_runner(const std::string& model_path) {
    g_ff_runner_instance = std::make_unique<FFNetONNXRunner>(model_path);
}

FFNetONNXRunner& get_ff_runner() {
    return *g_ff_runner_instance;
}

} // namespace ff_interface

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " model.onnx\n";
        return 1;
    }

    const std::string model_path = argv[1];

    try {
        ff_interface::initialize_ff_runner(model_path);
        auto& runner = ff_interface::get_ff_runner();

        // Dummy inputs

        // [ pt, eta, mass,
        //   seedingJet_pt, seedingJet_eta, seedingJet_mass,
        //   btagPNetB, btagPNetCvB, btagPNetCvL, btagPNetCvNotB, btagPNetQvG ]

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

        // decayMode
        int decayMode = 1;

        float w_ff = runner.compute_w_ff(cont_features, decayMode);

        std::cout << "\n=== SUMMARY ===\n";
        std::cout << "Model path:     " << model_path << "\n";
        std::cout << "Decay mode:     " << decayMode << "\n";
        std::cout << "w_ff:           " << w_ff << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[error] " << ex.what() << "\n";
        return 1;
    }
}
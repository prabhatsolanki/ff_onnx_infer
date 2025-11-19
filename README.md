# FF ONNX Inference (Python & C++)

Helpers to run the fake factor (FF) ONNX model from both Python and C++ with the same conventions.

## Model

The ONNX model is expected to:

- Take a single input tensor named: `raw_input`
- Produce a single output tensor named: `w_ff`

## Inputs

- Expects 12 features:

  `cont_features = [pt, eta, mass, seedingJet_pt, seedingJet_eta, seedingJet_mass, btagPNetB, btagPNetCvB, btagPNetCvL, btagPNetCvNotB, btagPNetQvG]`
  `decayMode` = scalar integer (e.g. `0, 1, 2, 10, 11`)

The code internally converts `decayMode` into one-hot `decayMode_*` entries.

## Files

- `run_inf.py` – Python ONNX Runtime wrapper with a small runner class and a dummy test.
- `run_inf.cc` – C++ ONNX Runtime wrapper (namespace `ff_interface`) plus a small test `main`.
- `models/` – exported FF model and feature order JSON file.

## Usage

**Python**

- Install: `onnxruntime`, `numpy`
- Example:  
  `python run_inf.py --onnx path/to/model.onnx [--config path/to/model.json]`

- To import FFNetONNXRunner in analysis:

   `from run_ff import FFNetONNXRunner`
  
   `runner = FFNetONNXRunner("model.onnx", "model.json")`
  
   `w = runner.compute_w_ff(features)`

**C++**

- Compile `run_inf.cc` against ONNX Runtime
  `g++ run_inf.cc -o ff_infer   -I/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/include/onnxruntime  -L/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/lib64 -lonnxruntime`
- Example:  
  `./ff_infer path/to/model.onnx path/to/model.json` 

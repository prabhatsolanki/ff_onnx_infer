# FF ONNX Inference (Python & C++)

Helpers to run the fake factor (FF) ONNX model from both Python and C++ with the same conventions.

## Model

The ONNX model is expected to:

- Take a single input tensor named: `raw_input`
- Produce a single output tensor named: `w_ff`
- Expect a 16-dimensional feature vector in this exact order:

1. `pt`  
2. `eta`  
3. `mass`  
4. `seedingJet_pt`  
5. `seedingJet_eta`  
6. `seedingJet_mass`  
7. `decayMode_0`  
8. `decayMode_1`  
9. `decayMode_2`  
10. `decayMode_10`  
11. `decayMode_11`  
12. `btagPNetB`  
13. `btagPNetCvB`  
14. `btagPNetCvL`  
15. `btagPNetCvNotB`  
16. `btagPNetQvG`  

In both implementations you provide:

- `cont_features = [pt, eta, mass, seedingJet_pt, seedingJet_eta, seedingJet_mass, btagPNetB, btagPNetCvB, btagPNetCvL, btagPNetCvNotB, btagPNetQvG]`
- `decayMode` = scalar integer (e.g. 0, 1, 2, 10, 11)

The code internally converts `decayMode` into one-hot `decayMode_*` entries (indices 6–10).

## Files

- `run_inf.py` – Python ONNX Runtime wrapper with a small runner class and a dummy test.
- `run_inf.cc` – C++ ONNX Runtime wrapper (namespace `ff_interface`) plus a small test `main`.
- `models/Run3_2022EE/model.onnx` – exported FF model matching the feature order above.

## Usage

**Python**

- Install: `onnxruntime`, `numpy`
- Example:  
  `python run_ff.py --onnx path/to/model.onnx`

**C++**

- Compile `run_inf.cc` against ONNX Runtime
  `g++ run_inf.cc -o ff_infer   -I/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/include   /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/lib64/libonnxruntime.so`
- Example:  
  `./ff_infer path/to/model.onnx`

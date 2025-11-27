# FF ONNX K-Fold Inference

Helpers to run a k-fold fake-factor (FF) ONNX setup from both Python and C++.

## Model layout

Each fold has:

- `model_fold{N}.onnx` – ONNX model with:
  - input tensor name: `raw_input`
  - output tensor name: `w_ff`
- `feature_order_fold{N}.json` – JSON with:
    {
      "feature_order": [
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
      ],
      "model_type": "single",
      "fold": 0
    }

`decayMode_*` entries are constructed internally from the scalar `decayMode`.

## Inputs

Per-event inputs (Python: NumPy arrays, C++: `std::vector<float>` / `std::vector<long long>`):

- `event_id` (integer, used as `event_id % n_folds` to select the fold)
- `decayMode` (integer: e.g. 0, 1, 2, 10, 11)
- `pt`, `eta`, `mass`
- `seedingJet_pt`, `seedingJet_eta`, `seedingJet_mass`
- `btagPNetB`, `btagPNetCvB`, `btagPNetCvL`, `btagPNetCvNotB`, `btagPNetQvG`

Feature names must match those in `feature_order_fold{N}.json`, but user-side ordering does not matter. The code:

- reads the JSON feature order for each fold,
- builds the ONNX input in that order,
- converts `decayMode` to one-hot `decayMode_*` internally.

## Files

- `run_inf_kfold.py` – Python ONNX Runtime wrapper (`KFoldFFONNX`) with a small example `main`.
- `run_inf_kfold.cc` – C++ ONNX Runtime wrapper (`KFoldFFONNX`) plus a small example `main`.
- `models/` – directory with `model_fold*.onnx` and `feature_order_fold*.json`.

## Usage

### Python

Requirements: `onnxruntime`, `numpy`

Run the example script:

    python run_inf_kfold.py --model-dir models

Use from analysis code:

    from run_inf_kfold import KFoldFFONNX
    import numpy as np

    runner = KFoldFFONNX("models")

    w = runner.compute_w_ff(
        event_id       = np.array([...], dtype=np.int64),
        decayMode      = np.array([...], dtype=np.int32),
        pt             = np.array([...], dtype=np.float32),
        eta            = np.array([...], dtype=np.float32),
        mass           = np.array([...], dtype=np.float32),
        seedingJet_pt  = np.array([...], dtype=np.float32),
        seedingJet_eta = np.array([...], dtype=np.float32),
        seedingJet_mass= np.array([...], dtype=np.float32),
        btagPNetB      = np.array([...], dtype=np.float32),
        btagPNetCvB    = np.array([...], dtype=np.float32),
        btagPNetCvL    = np.array([...], dtype=np.float32),
        btagPNetCvNotB = np.array([...], dtype=np.float32),
        btagPNetQvG    = np.array([...], dtype=np.float32),
    )

`w` is a 1D NumPy array of `w_ff` values, one per tau.

### C++

Compile (adjust include/library paths as needed):

    g++ run_inf_kfold.cc -o ff_infer \
        -I/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/include/onnxruntime \
        -L/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/lib64 \
        -lonnxruntime -std=c++17 -O2

Run:

    ./ff_infer models

and call:

   KFoldFFONNX kff("models");
   auto w = kff.compute_w_ff_event(event, decayMode,
                                    pt, eta, mass,
                                    seedingJet_pt, seedingJet_eta, seedingJet_mass,
                                    btagPNetB, btagPNetCvB, btagPNetCvL,
                                    btagPNetCvNotB, btagPNetQvG);
#!/usr/bin/env python3
"""
Usage:
  python run_inf.py --onnx models/model_v1911.onnx [--config models/model_v1911.json]

To import FFNetONNXRunner in analysis:

  from run_ff import FFNetONNXRunner
  runner = FFNetONNXRunner("model.onnx", "model.json")
  w = runner.compute_w_ff(features)
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union, List

import numpy as np
import onnxruntime as ort

LOGGER = logging.getLogger("FF_ONNX")

# Default ONNX feature order overridden by inout JSON config
DEFAULT_FEATURE_ORDER: List[str] = [
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
    "btagPNetQvG",
]

DEFAULT_CONT_FEATURE_NAMES: List[str] = [
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
    "btagPNetQvG",
]

ARRAY_WITH_DM_SIZE = len(DEFAULT_CONT_FEATURE_NAMES) + 1
ARRAY_DM_INDEX = 6  

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(message)s"
    )

class FFNetONNXRunner:
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find ONNX model file: {model_path}")

        self.feature_order = list(DEFAULT_FEATURE_ORDER)
        self.cont_feature_names = list(DEFAULT_CONT_FEATURE_NAMES)
        self.decay_mode_indices: List[tuple[int, int]] = []
        self.feature_index: dict[str, int] = {}

        self._load_config(model_path, config_path)
        self._init_feature_layout()

        LOGGER.info("Loading ONNX model from %s", model_path)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = "raw_input"
        self.output_name = "w_ff"

        LOGGER.info("ONNX session ready (features: %d)", len(self.feature_order))
        LOGGER.debug("feature_order: %s", self.feature_order)
        LOGGER.debug("cont_feature_names: %s", self.cont_feature_names)

    def _load_config(self, model_path: str, config_path: Optional[str]) -> None:
        if config_path:
            cfg_path = Path(config_path)
        else:
            cfg_path = Path(model_path).with_suffix(".json")

        if not cfg_path.exists():
            LOGGER.info("No config found at %s, using defaults.", cfg_path)
            return

        LOGGER.info("Loading input config from %s", cfg_path)
        data = json.loads(cfg_path.read_text())

        if not isinstance(data, dict):
            raise RuntimeError(f"Config file {cfg_path} must contain a JSON object.")

        if "feature_order" in data:
            self.feature_order = [str(x) for x in data["feature_order"]]

        if "cont_feature_names" in data:
            self.cont_feature_names = [str(x) for x in data["cont_feature_names"]]

        LOGGER.info(
            "Config loaded: feature_order=%d, cont_feature_names=%d",
            len(self.feature_order),
            len(self.cont_feature_names),
        )

    def _init_feature_layout(self) -> None:
        self.feature_index = {name: i for i, name in enumerate(self.feature_order)}
        self.decay_mode_indices = []
        for idx, name in enumerate(self.feature_order):
            if name.startswith("decayMode_"):
                try:
                    dm_val = int(name.split("_", 1)[1])
                    self.decay_mode_indices.append((dm_val, idx))
                except ValueError:
                    LOGGER.warning("Could not parse decayMode from '%s'", name)

        if not self.decay_mode_indices:
            LOGGER.warning("No decayMode_* entries found in feature_order")
        else:
            LOGGER.debug("decayMode layout: %s", self.decay_mode_indices)
        missing = [n for n in self.cont_feature_names if n not in self.feature_index]
        if missing:
            LOGGER.warning(
                "Some cont_feature_names not present in feature_order: %s", missing
            )

    def _from_mapping(
        self,
        features: Mapping[str, float],
        decay_mode: Optional[int],
    ) -> tuple[np.ndarray, int]:
        if decay_mode is None:
            if "decayMode" not in features:
                raise ValueError(
                    "Mapping input must contain 'decayMode' if no decay_mode argument is given."
                )
            dm = int(features["decayMode"])
        else:
            dm = int(decay_mode)

        cf = np.zeros(len(self.cont_feature_names), dtype=np.float32)
        for i, name in enumerate(self.cont_feature_names):
            if name in features:
                cf[i] = float(features[name])
        return cf, dm

    def _from_array(
        self,
        arr_like: Union[Sequence[float], np.ndarray],
        decay_mode: Optional[int],
    ) -> tuple[np.ndarray, int]:
        arr = np.asarray(arr_like, dtype=np.float32).ravel()
        if decay_mode is None:
            if arr.size != ARRAY_WITH_DM_SIZE:
                raise ValueError(
                    f"Array input (without explicit decay_mode) must have length "
                    f"{ARRAY_WITH_DM_SIZE} (cont + decayMode), got {arr.size}."
                )
            dm = int(round(float(arr[ARRAY_DM_INDEX])))

            cf = np.zeros(len(self.cont_feature_names), dtype=np.float32)
            cf[0:6] = arr[0:6]
            cf[6:] = arr[ARRAY_DM_INDEX + 1 :]
        else:
            dm = int(decay_mode)
            if arr.size != len(self.cont_feature_names):
                raise ValueError(
                    f"Array input (with explicit decay_mode) must have length "
                    f"{len(self.cont_feature_names)}, got {arr.size}."
                )
            cf = arr.copy()

        return cf, dm

    def _build_input_tensor(self, cf: np.ndarray, decay_mode: int) -> np.ndarray:
        x = np.zeros((1, len(self.feature_order)), dtype=np.float32)
        for i, name in enumerate(self.cont_feature_names):
            idx = self.feature_index.get(name)
            if idx is not None:
                x[0, idx] = cf[i]
        if not self.decay_mode_indices:
            LOGGER.warning("No decayMode_* features in feature_order, decayMode ignored.")
        else:
            matched = False
            for dm_val, idx in self.decay_mode_indices:
                if idx >= x.shape[1]:
                    continue
                x[0, idx] = 1.0 if dm_val == decay_mode else 0.0
                if dm_val == decay_mode:
                    matched = True
            if not matched:
                LOGGER.warning(
                    "DecayMode %d not among configured decayMode_* values; one-hot will be all zeros.",
                    decay_mode,
                )

        LOGGER.debug("Input tensor row: %s", x[0])
        return x

    def compute_w_ff(
        self,
        features: Union[Mapping[str, float], Sequence[float], np.ndarray],
        decay_mode: Optional[int] = None,
    ) -> float:
        if isinstance(features, Mapping):
            cf, dm = self._from_mapping(features, decay_mode)
        else:
            cf, dm = self._from_array(features, decay_mode)

        LOGGER.info("Running inference (decayMode=%d)", dm)
        x = self._build_input_tensor(cf, dm)

        out = self.session.run([self.output_name], {self.input_name: x})[0]
        out_flat = np.asarray(out, dtype=np.float32).ravel()
        if out_flat.size == 0:
            raise RuntimeError("Model output is empty.")
        return float(out_flat[0])


_runner_instance: Optional[FFNetONNXRunner] = None

def initialize_ff_runner(model_path: str, config_path: Optional[str] = None) -> None:
    global _runner_instance
    _runner_instance = FFNetONNXRunner(model_path, config_path)

def get_ff_runner() -> FFNetONNXRunner:
    if _runner_instance is None:
        raise RuntimeError("FFNetONNXRunner has not been initialized.")
    return _runner_instance

def main() -> None:
    ap = argparse.ArgumentParser("FF ONNX inference")
    ap.add_argument("--onnx", required=True, help="Path to model.onnx")
    ap.add_argument("--config", help="Optional JSON config with feature_order")
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logs")
    args = ap.parse_args()

    setup_logging(args.verbose)

    runner = FFNetONNXRunner(args.onnx, args.config)

    # Example 1: dict with names
    features_dict = {
        "pt": 45.0,
        "eta": 0.3,
        "mass": 1.2,
        "seedingJet_pt": 50.0,
        "seedingJet_eta": 0.1,
        "seedingJet_mass": 10.0,
        "btagPNetB": 0.2,
        "btagPNetCvB": 0.1,
        "btagPNetCvL": 0.4,
        "btagPNetCvNotB": 0.3,
        "btagPNetQvG": 0.5,
        "decayMode": 1,
    }
    w_dict = runner.compute_w_ff(features_dict)

    # Example 2: plain numpy array 
    features_arr = np.array(
        [
            45.0,  # pt
            0.3,   # eta
            1.2,   # mass
            50.0,  # seedingJet_pt
            0.1,   # seedingJet_eta
            10.0,  # seedingJet_mass
            1.0,   # decayMode 
            0.2,   # btagPNetB
            0.1,   # btagPNetCvB
            0.4,   # btagPNetCvL
            0.3,   # btagPNetCvNotB
            0.5,   # btagPNetQvG
        ],
        dtype=np.float32,
    )
    w_arr = runner.compute_w_ff(features_arr)

    print("\n=== SUMMARY ===")
    print(f"ONNX model:   {args.onnx}")
    if args.config:
        print(f"Config:       {args.config}")
    print(f"feature_order length: {len(runner.feature_order)}")
    print(f"w_ff (dict):  {w_dict:.6g}")
    print(f"w_ff (array): {w_arr:.6g}")


if __name__ == "__main__":
    main()
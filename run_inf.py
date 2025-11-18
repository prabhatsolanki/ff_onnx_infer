#!/usr/bin/env python3
"""
Usage:
  python run_ff.py --onnx models/Run3_2022EE/model.onnx
"""

import argparse
import logging
import os
from typing import List, Optional

import numpy as np
import onnxruntime as ort

LOGGER = logging.getLogger("FF_ONNX")

FEATURE_ORDER: List[str] = [
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

DECAY_MODES_TO_ENCODE: List[int] = [0, 1, 2, 10, 11]
DECAY_MODE: int = 1


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


class FFNetONNXRunner:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find ONNX model file: {model_path}")

        LOGGER.info("Loading ONNX model from %s", model_path)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = "raw_input"
        self.output_name = "w_ff"

        LOGGER.info("ONNX session created.")
        LOGGER.info("Feature order (N=%d):", len(FEATURE_ORDER))
        for i, name in enumerate(FEATURE_ORDER):
            LOGGER.info("  [%02d] %s", i, name)

    def _encode_decay_mode_one_hot(self, decay_mode: int) -> np.ndarray:
        one_hot = np.zeros(len(DECAY_MODES_TO_ENCODE), dtype=np.float32)
        matched = False
        matched_dm = None

        for i, dm in enumerate(DECAY_MODES_TO_ENCODE):
            if dm == decay_mode:
                one_hot[i] = 1.0
                matched = True
                matched_dm = dm
            LOGGER.debug(
                "decayMode one hot: dm=%d -> %s", dm, one_hot[i]
            )

        if not matched:
            LOGGER.warning(
                "DecayMode %d not in %s; all decayMode_* entries will be 0.",
                decay_mode,
                DECAY_MODES_TO_ENCODE,
            )
        else:
            LOGGER.info("Encoded decayMode=%d as one hot.", matched_dm)

        return one_hot

    def compute_w_ff(self, cont_features: List[float], decay_mode: int) -> float:
        if len(cont_features) != 11:
            LOGGER.warning(
                "cont_features has length %d (expected 11: "
                "pt,eta,mass,jet_pt,jet_eta,jet_mass,B,CvB,CvL,CvNotB,QvG)",
                len(cont_features),
            )

        cf = np.asarray(cont_features, dtype=np.float32)
        LOGGER.info(
            "cont_features (len=%d): %s",
            len(cf),
            np.array2string(cf, precision=3, floatmode="fixed"),
        )
        LOGGER.info("decayMode = %d", decay_mode)

        x = np.zeros((1, len(FEATURE_ORDER)), dtype=np.float32)

        if len(cf) >= 6:
            x[0, 0] = cf[0]  # pt
            x[0, 1] = cf[1]  # eta
            x[0, 2] = cf[2]  # mass
            x[0, 3] = cf[3]  # seedingJet_pt
            x[0, 4] = cf[4]  # seedingJet_eta
            x[0, 5] = cf[5]  # seedingJet_mass

        x[0, 6:11] = self._encode_decay_mode_one_hot(decay_mode)

        if len(cf) >= 11:
            x[0, 11] = cf[6]   # btagPNetB
            x[0, 12] = cf[7]   # btagPNetCvB
            x[0, 13] = cf[8]   # btagPNetCvL
            x[0, 14] = cf[9]   # btagPNetCvNotB
            x[0, 15] = cf[10]  # btagPNetQvG

        LOGGER.info("Input tensor shape: %s, dtype: %s", x.shape, x.dtype)
        LOGGER.info("Features (name -> value):")
        for name, val in list(zip(FEATURE_ORDER, x[0])):
            LOGGER.info("  %-20s = % .6g", name, val)

        LOGGER.debug("Full input vector: %s", x)

        LOGGER.info("Running ONNX inference...")
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        LOGGER.info("Raw output shape: %s, dtype: %s", out.shape, out.dtype)
        LOGGER.debug("Raw output values: %s", out)

        out_flat = np.asarray(out, dtype=np.float32).reshape(-1)
        if out_flat.size == 0:
            raise RuntimeError("Model output is empty.")
        w_ff = float(out_flat[0])
        LOGGER.info("w_ff = %.6g", w_ff)
        return w_ff


_runner_instance: Optional[FFNetONNXRunner] = None


def initialize_ff_runner(model_path: str) -> None:
    global _runner_instance
    _runner_instance = FFNetONNXRunner(model_path)


def get_ff_runner() -> FFNetONNXRunner:
    if _runner_instance is None:
        raise RuntimeError("FFNetONNXRunner has not been initialized.")
    return _runner_instance


def build_dummy_cont_features() -> List[float]:
    cont_features = [
        45.0,  # pt
        0.3,   # eta
        1.2,   # mass
        50.0,  # seedingJet_pt
        0.1,   # seedingJet_eta
        10.0,  # seedingJet_mass
        0.2,   # btagPNetB
        0.1,   # btagPNetCvB
        0.4,   # btagPNetCvL
        0.3,   # btagPNetCvNotB
        0.5,   # btagPNetQvG
    ]
    LOGGER.info("Dummy cont_features: %s", cont_features)
    LOGGER.info("Dummy decayMode = %d", DECAY_MODE)
    return cont_features


def main():
    ap = argparse.ArgumentParser("FF ONNX inference")
    ap.add_argument("--onnx", required=True, help="Path to model.onnx")
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logs")
    args = ap.parse_args()

    setup_logging(args.verbose)

    initialize_ff_runner(args.onnx)
    runner = get_ff_runner()

    cont_features = build_dummy_cont_features()
    w_ff = runner.compute_w_ff(cont_features, DECAY_MODE)

    print("\n=== SUMMARY ===")
    print(f"ONNX model:    {args.onnx}")
    print(f"num features:  {len(FEATURE_ORDER)}")
    print(f"Decay mode:    {DECAY_MODE}")
    print(f"w_ff:          {w_ff:.6g}")


if __name__ == "__main__":
    main()
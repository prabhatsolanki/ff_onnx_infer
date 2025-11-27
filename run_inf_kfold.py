#!/usr/bin/env python3
"""
K-fold FF ONNX inference

Usage:
  python run_inf_kfold.py --model-dir models
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort

LOG = logging.getLogger("FF_KFOLD")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


class KFoldFFONNXRunner:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)

        folds: List[int] = []
        for path in self.model_dir.glob("feature_order_fold*.json"):
            stem = path.stem 
            if "fold" not in stem:
                continue
            try:
                idx = int(stem.split("fold", 1)[1])
                folds.append(idx)
            except ValueError:
                continue

        if not folds:
            raise RuntimeError(f"No feature_order_fold*.json found in {self.model_dir}")

        self.n_folds = max(folds) + 1
        LOG.info("Initialising K-fold runner from %s (n_folds=%d)", self.model_dir, self.n_folds)

        self.sessions: List[ort.InferenceSession] = []
        self.fold_feature_order: List[List[str]] = []
        self.fold_feature_index: List[Dict[str, int]] = []
        self.fold_dm_values: List[List[int]] = []
        self.fold_dm_indices: List[List[int]] = []

        for fold in range(self.n_folds):
            onnx_path = self.model_dir / f"model_fold{fold}.onnx"
            json_path = self.model_dir / f"feature_order_fold{fold}.json"

            if not onnx_path.exists():
                raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")
            if not json_path.exists():
                raise FileNotFoundError(f"Missing feature-order JSON: {json_path}")

            data = json.loads(json_path.read_text())
            feature_order = list(data["feature_order"])
            self.fold_feature_order.append(feature_order)

            feature_index = {name: i for i, name in enumerate(feature_order)}
            self.fold_feature_index.append(feature_index)

            dm_vals: List[int] = []
            dm_idx: List[int] = []
            for i, name in enumerate(feature_order):
                if name.startswith("decayMode_"):
                    dm_vals.append(int(name.split("_", 1)[1]))
                    dm_idx.append(i)
            self.fold_dm_values.append(dm_vals)
            self.fold_dm_indices.append(dm_idx)

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self.sessions.append(sess)

            LOG.info("Fold %d: ONNX initialised from %s", fold, onnx_path)

        base_order = self.fold_feature_order[0]
        self.scalar_feature_names = [n for n in base_order if not n.startswith("decayMode_")]
        self.decay_feature_names = [n for n in base_order if n.startswith("decayMode_")]

        LOG.info("Feature order (fold 0): %s", ", ".join(base_order))

    def compute_w_ff(self, event_id: np.ndarray, **feat_arrays: np.ndarray) -> np.ndarray:
        event_id = np.asarray(event_id, dtype=np.int64).ravel()
        n = event_id.size
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        if "decayMode" not in feat_arrays:
            raise ValueError("Missing required feature 'decayMode'")

        decay_mode = np.asarray(feat_arrays["decayMode"], dtype=np.int32).ravel()
        if decay_mode.size != n:
            raise ValueError("decayMode length mismatch with event_id")

        LOG.info("compute_w_ff: n_taus=%d", n)
        LOG.debug("User features: %s", ", ".join(sorted(feat_arrays.keys())))
        LOG.debug("Scalar features: %s", ", ".join(self.scalar_feature_names))
        LOG.debug("Decay one-hot features: %s", ", ".join(self.decay_feature_names))

        scalar_arrays: Dict[str, np.ndarray] = {}
        for name in self.scalar_feature_names:
            if name not in feat_arrays:
                raise ValueError(f"Missing required feature '{name}'")
            arr = np.asarray(feat_arrays[name], dtype=np.float32).ravel()
            if arr.size != n:
                raise ValueError(f"Feature '{name}' has length {arr.size}, expected {n}")
            scalar_arrays[name] = arr

        w = np.zeros(n, dtype=np.float32)

        for fold in range(self.n_folds):
            mask = (event_id % self.n_folds) == fold
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                continue

            fo = self.fold_feature_order[fold]
            fi = self.fold_feature_index[fold]
            dm_vals = self.fold_dm_values[fold]
            dm_idx = self.fold_dm_indices[fold]

            m = idxs.size
            n_feat = len(fo)
            x = np.zeros((m, n_feat), dtype=np.float32)

            for name, arr in scalar_arrays.items():
                j = fi.get(name)
                if j is None:
                    continue
                x[:, j] = arr[idxs]

            # decayMode one-hot
            dm_slice = decay_mode[idxs]
            for dv, col in zip(dm_vals, dm_idx):
                x[:, col] = (dm_slice == dv).astype(np.float32)

            sess = self.sessions[fold]
            y = sess.run(["w_ff"], {"raw_input": x})[0].reshape(-1).astype(np.float32)
            w[idxs] = y

        return w


def main() -> None:
    ap = argparse.ArgumentParser("K-fold FF ONNX inference")
    ap.add_argument(
        "--model-dir",
        required=True,
        help="Directory with model_fold*.onnx and feature_order_fold*.json",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    setup_logging(args.verbose)

    runner = KFoldFFONNXRunner(args.model_dir)

    # example
    n = 8
    event_id   = np.array([1001, 1001, 1002, 1003, 1004, 1005, 1005, 1006], dtype=np.int64)
    decay_mode = np.array([0, 1, 2, 10, 11, 0, 1, 2], dtype=np.int32)

    def fill(v: float) -> np.ndarray:
        return np.full(n, v, dtype=np.float32)

    pt             = fill(45.0)
    eta            = fill(0.3)
    mass           = fill(1.2)
    seedingJet_pt  = fill(50.0)
    seedingJet_eta = fill(0.1)
    seedingJet_mass= fill(10.0)
    btagPNetB      = fill(0.2)
    btagPNetCvB    = fill(0.1)
    btagPNetCvL    = fill(0.4)
    btagPNetCvNotB = fill(0.3)
    btagPNetQvG    = fill(0.5)

    w = runner.compute_w_ff(
        event_id=event_id,
        decayMode=decay_mode,
        pt=pt,
        eta=eta,
        mass=mass,
        seedingJet_pt=seedingJet_pt,
        seedingJet_eta=seedingJet_eta,
        seedingJet_mass=seedingJet_mass,
        btagPNetB=btagPNetB,
        btagPNetCvB=btagPNetCvB,
        btagPNetCvL=btagPNetCvL,
        btagPNetCvNotB=btagPNetCvNotB,
        btagPNetQvG=btagPNetQvG,
    )

    print("event_id  fold  decayMode  w_ff")
    for i in range(n):
        fold = int(event_id[i] % runner.n_folds)
        print(f"{int(event_id[i]):7d}  {fold:4d}  {int(decay_mode[i]):9d}  {w[i]:.6g}")


if __name__ == "__main__":
    main()
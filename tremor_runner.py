"""Inference runner for the saved tremor ensemble."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tremor_features import (
    SENSOR_COLUMNS,
    WINDOW,
    STRIDE,
    SEGMENT_LEN,
    segment_windows,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tremor_model" / "tremor_ensemble.pkl"
CONFIG_PATH = BASE_DIR / "tremor_model" / "tremor_model_config.json"
DATA_PATH = BASE_DIR / "tremor_data" / "Dataset.csv"


class TremorModelRunner:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Tremor model missing: {MODEL_PATH}")
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Tremor config missing: {CONFIG_PATH}")
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Tremor dataset missing: {DATA_PATH}")

        self.config = json.loads(CONFIG_PATH.read_text())
        self.bundle = joblib.load(MODEL_PATH)
        self.n_segments = int(self.bundle["n_segments"])
        self.threshold = float(self.bundle.get("threshold", 0.5))

        self.df = pd.read_csv(DATA_PATH)
        missing = [c for c in SENSOR_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Dataset is missing sensor columns: {missing}")

        self.sensor_data = self.df[SENSOR_COLUMNS].to_numpy(dtype=float)
        self.labels = (
            self.df["Result"].to_numpy()
            if "Result" in self.df.columns
            else None
        )

        self.folds = self.bundle["fold_models"]
        self.seg_edges = self.bundle["seg_edges"]

    def _fold_for_segment(self, segment_index):
        """Choose the held-out fold corresponding to this dataset segment."""
        for fold_index in range(len(self.seg_edges) - 1):
            if (
                self.seg_edges[fold_index]
                <= segment_index
                < self.seg_edges[fold_index + 1]
            ):
                return self.folds[fold_index]
        raise ValueError("Could not find a model fold for this segment.")

    def _predict_array(self, data, models):
        """Predict sensor data and average window probabilities."""
        windows = segment_windows(data)
        if len(windows) == 0:
            raise ValueError(
                f"At least {WINDOW} valid sensor rows are required."
            )

        model_names = {
            "extra_trees": "extra_trees",
            "random_forest": "random_forest",
            "gradient_boosting": "hist_gb",
        }
        scores = {}

        for output_name, model_name in model_names.items():
            model = models[model_name]
            probabilities = model.predict_proba(windows)[:, 1]
            scores[output_name] = float(np.mean(probabilities))

        combined = float(np.mean(list(scores.values())))
        predicted_class = int(combined >= self.threshold)

        return {
            "extra_trees_score": scores["extra_trees"],
            "random_forest_score": scores["random_forest"],
            "gradient_boosting_score": scores["gradient_boosting"],
            "combined_score": combined,
            "predicted_class": predicted_class,
            "predicted_label": (
                "Parkinson's (Tremor Pattern)"
                if predicted_class == 1
                else "Healthy / Control (Tremor Pattern)"
            ),
            "windows_analysed": int(len(windows)),
        }

    def predict_segment(self, segment):
        segment = int(segment)
        if not 1 <= segment <= self.n_segments:
            raise ValueError(
                f"Segment must be between 1 and {self.n_segments}."
            )

        start = (segment - 1) * SEGMENT_LEN
        end = start + SEGMENT_LEN
        data = self.sensor_data[start:end]

        if len(data) < SEGMENT_LEN:
            raise ValueError(f"Dataset segment {segment} is incomplete.")

        models = self._fold_for_segment(segment - 1)
        result = self._predict_array(data, models)

        true_label = None
        if self.labels is not None:
            true_label = int(
                np.mean(self.labels[start:end].astype(float)) >= 0.5
            )

        result.update({
            "source": "Benchmark Segment",
            "dataset_number": segment,
            "rows": int(len(data)),
            "dataset_true_label": true_label,
            "match": (
                result["predicted_class"] == true_label
                if true_label is not None
                else None
            ),
        })
        return result

    def predict_csv(self, csv_path, source_name=None):
        df = pd.read_csv(csv_path)
        missing = [c for c in SENSOR_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                "Uploaded CSV must contain these sensor columns: "
                + ", ".join(SENSOR_COLUMNS)
            )

        data = df[SENSOR_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(data).all():
            raise ValueError("Uploaded sensor data contains missing or invalid values.")

        # Uploaded data has no known held-out segment. Use the ensemble
        # of saved fold models and average their probabilities.
        fold_results = [
            self._predict_array(data, models)
            for models in self.folds
        ]

        score_keys = [
            "extra_trees_score",
            "random_forest_score",
            "gradient_boosting_score",
            "combined_score",
        ]
        result = {
            key: float(np.mean([r[key] for r in fold_results]))
            for key in score_keys
        }
        result["predicted_class"] = int(
            result["combined_score"] >= self.threshold
        )
        result["predicted_label"] = (
            "Parkinson's (Tremor Pattern)"
            if result["predicted_class"] == 1
            else "Healthy / Control (Tremor Pattern)"
        )
        result["windows_analysed"] = fold_results[0]["windows_analysed"]
        result.update({
            "source": source_name or "Uploaded CSV",
            "dataset_number": None,
            "rows": int(len(data)),
            "dataset_true_label": None,
            "match": None,
        })
        return result

    def get_segments_catalog(self):
        return {
            "n_segments": self.n_segments,
            "segment_len": SEGMENT_LEN,
            "segments": [
                {"segment": i, "label": f"Dataset Segment {i}"}
                for i in range(1, self.n_segments + 1)
            ],
        }

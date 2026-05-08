"""
model.py — LightGBM binary classifier for candle direction prediction.
Handles training, evaluation, persistence, and inference.
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError(
        "lightgbm is required. Install with: pip install lightgbm"
    )

from utils import get_logger

logger = get_logger("Model")


class DirectionModel:
    """
    Binary classifier that predicts whether price will be higher
    (class=1, UP) or lower (class=0, DOWN) after N candles.

    Wraps LightGBM with train / save / load / predict interface.
    """

    def __init__(self, cfg: Dict[str, Any]):
        model_cfg = cfg["model"]
        self.model_path: str = model_cfg["path"]
        self.feature_cols: List[str] = model_cfg["features"]
        self.lookahead: int = model_cfg["target_lookahead"]
        self.train_test_split: float = model_cfg["train_test_split"]
        self.lgbm_params: Dict = model_cfg.get("lgbm_params", {})

        self._model: Optional[lgb.LGBMClassifier] = None

    # ─────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train LightGBM on (X, y).
        Uses a chronological train/test split (no shuffle).
        Returns evaluation metrics dict.
        """
        n = len(X)
        split_idx = int(n * self.train_test_split)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(
            f"Training on {len(X_train)} samples, testing on {len(X_test)} samples"
        )

        # Build LightGBM params from config
        params = {
            "n_estimators": self.lgbm_params.get("n_estimators", 300),
            "learning_rate": self.lgbm_params.get("learning_rate", 0.05),
            "max_depth": self.lgbm_params.get("max_depth", 6),
            "num_leaves": self.lgbm_params.get("num_leaves", 31),
            "min_child_samples": self.lgbm_params.get("min_child_samples", 20),
            "subsample": self.lgbm_params.get("subsample", 0.8),
            "colsample_bytree": self.lgbm_params.get("colsample_bytree", 0.8),
            "class_weight": self.lgbm_params.get("class_weight", "balanced"),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }

        self._model = lgb.LGBMClassifier(**params)

        # Fit with early stopping on eval set
        eval_set = [(X_test.values, y_test.values)]
        self._model.fit(
            X_train.values,
            y_train.values,
            eval_set=eval_set,
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        # Evaluate
        metrics = self._evaluate(X_test, y_test)
        self._log_feature_importance()
        return metrics

    def _evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Compute and log test-set metrics."""
        y_pred = self._model.predict(X_test.values)
        y_prob = self._model.predict_proba(X_test.values)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        logger.info(f"Test Accuracy: {acc:.4f}  |  AUC-ROC: {auc:.4f}")
        logger.info(
            "\n" + classification_report(y_test, y_pred, target_names=["DOWN", "UP"])
        )
        return {"accuracy": acc, "auc_roc": auc}

    def _log_feature_importance(self) -> None:
        """Log top-10 feature importances."""
        importances = self._model.feature_importances_
        ranked = sorted(
            zip(self.feature_cols, importances), key=lambda x: -x[1]
        )
        lines = ["Feature Importances (top 10):"]
        for feat, imp in ranked[:10]:
            lines.append(f"  {feat:<20} {imp:.0f}")
        logger.info("\n".join(lines))

    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    def save(self, path: str = None) -> None:
        """Pickle model to disk."""
        path = path or self.model_path
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str = None) -> bool:
        """
        Load model from disk.
        Returns True if successful, False if file does not exist.
        """
        path = path or self.model_path
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return True

    def is_trained(self) -> bool:
        return self._model is not None

    # ─────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> Tuple[float, float]:
        """
        Return (prob_down, prob_up) for the given feature vector.
        X should be a single-row DataFrame with self.feature_cols columns.
        """
        if not self.is_trained():
            raise RuntimeError("Model is not trained or loaded.")

        proba = self._model.predict_proba(X.values)   # shape (1, 2)
        prob_down = float(proba[0, 0])
        prob_up = float(proba[0, 1])
        return prob_down, prob_up

    def predict_direction(
        self, X: pd.DataFrame, threshold: float = 0.5
    ) -> Tuple[str, float]:
        """
        Return (direction, confidence) where:
          direction ∈ {'UP', 'DOWN'}
          confidence is the probability of that direction.
        """
        prob_down, prob_up = self.predict_proba(X)
        if prob_up >= threshold:
            return "UP", prob_up
        else:
            return "DOWN", prob_down

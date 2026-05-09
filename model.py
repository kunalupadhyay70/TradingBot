"""
model.py — LightGBM binary classifier for candle direction prediction.
Handles training, evaluation, persistence, and inference.

Enhancements:
- Feature scaling with StandardScaler
- Time series cross-validation
- Enhanced evaluation metrics (Precision, Recall, F1, Specificity)
- Better hyperparameter tuning
- Top-15 feature importance
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

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
    Includes feature scaling and cross-validation.
    """

    def __init__(self, cfg: Dict[str, Any]):
        model_cfg = cfg["model"]
        self.model_path: str = model_cfg["path"]
        self.feature_cols: List[str] = model_cfg["features"]
        self.lookahead: int = model_cfg["target_lookahead"]
        self.train_test_split: float = model_cfg["train_test_split"]
        self.lgbm_params: Dict = model_cfg.get("lgbm_params", {})
        self.use_scaling: bool = model_cfg.get("use_feature_scaling", True)
        self.use_cv: bool = model_cfg.get("use_cross_validation", True)
        self.cv_splits: int = model_cfg.get("cv_splits", 5)

        self._model: Optional[lgb.LGBMClassifier] = None
        self._scaler: Optional[StandardScaler] = None

    # ─────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train LightGBM on (X, y).
        Uses chronological train/test split and time series cross-validation.
        Returns evaluation metrics dict.
        """
        n = len(X)
        split_idx = int(n * self.train_test_split)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(
            f"Training on {len(X_train)} samples, testing on {len(X_test)} samples"
        )

        # ── Feature Scaling ────────────────────
        if self.use_scaling:
            self._scaler = StandardScaler()
            X_train_scaled = self._scaler.fit_transform(X_train.values)
            X_test_scaled = self._scaler.transform(X_test.values)
            logger.info("Feature scaling applied")
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        # Build LightGBM params from config
        params = {
            "n_estimators": self.lgbm_params.get("n_estimators", 500),
            "learning_rate": self.lgbm_params.get("learning_rate", 0.03),
            "max_depth": self.lgbm_params.get("max_depth", 7),
            "num_leaves": self.lgbm_params.get("num_leaves", 63),
            "min_child_samples": self.lgbm_params.get("min_child_samples", 20),
            "subsample": self.lgbm_params.get("subsample", 0.7),
            "colsample_bytree": self.lgbm_params.get("colsample_bytree", 0.7),
            "class_weight": self.lgbm_params.get("class_weight", "balanced"),
            "lambda_l1": self.lgbm_params.get("lambda_l1", 0.1),
            "lambda_l2": self.lgbm_params.get("lambda_l2", 0.1),
            "min_split_gain": self.lgbm_params.get("min_split_gain", 0.01),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }

        self._model = lgb.LGBMClassifier(**params)

        # Fit with early stopping on eval set
        eval_set = [(X_test_scaled, y_test.values)]
        self._model.fit(
            X_train_scaled,
            y_train.values,
            eval_set=eval_set,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        # ── Evaluate on test set ────────────────
        metrics = self._evaluate(X_test_scaled, y_test)

        # ── Cross-validation ────────────────────
        if self.use_cv:
            cv_metrics = self._cross_validate(X_train.values, y_train.values)
            metrics["cv_accuracy_mean"] = cv_metrics["accuracy_mean"]
            metrics["cv_accuracy_std"] = cv_metrics["accuracy_std"]
            metrics["cv_f1_mean"] = cv_metrics["f1_mean"]

        self._log_feature_importance()
        return metrics

    def _evaluate(self, X_test: np.ndarray, y_test: pd.Series) -> Dict[str, float]:
        """Compute comprehensive test-set metrics."""
        y_pred = self._model.predict(X_test)
        y_prob = self._model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        logger.info(
            f"\nTest Metrics:"
            f"\n  Accuracy:   {acc:.4f}"
            f"\n  AUC-ROC:    {auc:.4f}"
            f"\n  Precision:  {precision:.4f}"
            f"\n  Recall:     {recall:.4f}"
            f"\n  F1-Score:   {f1:.4f}"
            f"\n  Specificity: {specificity:.4f}"
        )

        logger.info("\n" + classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

        return {
            "accuracy": acc,
            "auc_roc": auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
        }

    def _cross_validate(self, X: np.ndarray, y: pd.Series) -> Dict[str, float]:
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        accuracies = []
        f1_scores = []

        fold = 0
        for train_idx, test_idx in tscv.split(X):
            fold += 1
            X_train_cv, X_test_cv = X[train_idx], X[test_idx]
            y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

            # Scale
            if self.use_scaling:
                scaler = StandardScaler()
                X_train_cv = scaler.fit_transform(X_train_cv)
                X_test_cv = scaler.transform(X_test_cv)

            # Train
            cv_model = lgb.LGBMClassifier(
                n_estimators=self.lgbm_params.get("n_estimators", 300),
                learning_rate=self.lgbm_params.get("learning_rate", 0.03),
                max_depth=self.lgbm_params.get("max_depth", 7),
                num_leaves=self.lgbm_params.get("num_leaves", 63),
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
            cv_model.fit(X_train_cv, y_train_cv.values)

            # Evaluate
            y_pred_cv = cv_model.predict(X_test_cv)
            acc = accuracy_score(y_test_cv, y_pred_cv)
            f1 = f1_score(y_test_cv, y_pred_cv, zero_division=0)

            accuracies.append(acc)
            f1_scores.append(f1)

            logger.info(f"  Fold {fold}: Accuracy={acc:.4f}, F1={f1:.4f}")

        logger.info(
            f"\nCross-Validation Results ({self.cv_splits} folds):"
            f"\n  Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}"
            f"\n  Mean F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}"
        )

        return {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "f1_mean": float(np.mean(f1_scores)),
        }

    def _log_feature_importance(self) -> None:
        """Log top-15 feature importances."""
        importances = self._model.feature_importances_
        ranked = sorted(
            zip(self.feature_cols, importances), key=lambda x: -x[1]
        )
        lines = ["\nFeature Importances (top 15):"]
        for i, (feat, imp) in enumerate(ranked[:15], 1):
            lines.append(f"  {i:2d}. {feat:<20} {imp:>8.0f}")
        logger.info("\n".join(lines))

    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    def save(self, path: str = None) -> None:
        """Pickle model and scaler to disk."""
        path = path or self.model_path
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "scaler": self._scaler}, f)
        logger.info(f"Model and scaler saved to {path}")

    def load(self, path: str = None) -> bool:
        """
        Load model and scaler from disk.
        Returns True if successful, False if file does not exist.
        """
        path = path or self.model_path
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
            self._model = data.get("model")
            self._scaler = data.get("scaler")
        logger.info(f"Model and scaler loaded from {path}")
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

        X_scaled = X.values
        if self.use_scaling and self._scaler is not None:
            X_scaled = self._scaler.transform(X_scaled)

        proba = self._model.predict_proba(X_scaled)  # shape (1, 2)
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

"""
model.py — LightGBM binary classifier with advanced ML techniques.
Includes feature scaling, time-series cross-validation, and enhanced metrics.

ENHANCED VERSION: 
- Feature Scaling (StandardScaler)
- Time Series Cross-Validation
- Better hyperparameters
- Enhanced evaluation metrics
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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

    ENHANCEMENTS:
    - Feature scaling with StandardScaler
    - Time Series Cross-Validation (5-fold)
    - Enhanced evaluation metrics
    - Better hyperparameter tuning
    """

    def __init__(self, cfg: Dict[str, Any]):
        model_cfg = cfg["model"]
        self.model_path: str = model_cfg["path"]
        self.feature_cols: List[str] = model_cfg["features"]
        self.lookahead: int = model_cfg["target_lookahead"]
        self.train_test_split: float = model_cfg["train_test_split"]
        self.lgbm_params: Dict = model_cfg.get("lgbm_params", {})
        
        # NEW: Scaling and cross-validation flags
        self.use_feature_scaling: bool = model_cfg.get("use_feature_scaling", True)
        self.use_cross_validation: bool = model_cfg.get("use_cross_validation", True)
        self.cv_splits: int = model_cfg.get("cv_splits", 5)

        self._model: Optional[lgb.LGBMClassifier] = None
        self._scaler: Optional[StandardScaler] = None

    # ─────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train LightGBM on (X, y) with enhancements.
        Uses chronological train/test split with cross-validation.
        Returns evaluation metrics dict.
        """
        n = len(X)
        split_idx = int(n * self.train_test_split)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(
            f"Training on {len(X_train)} samples, testing on {len(X_test)} samples"
        )

        # ── Feature Scaling ──────────────────────
        if self.use_feature_scaling:
            self._scaler = StandardScaler()
            X_train_scaled = self._scaler.fit_transform(X_train.values)
            X_test_scaled = self._scaler.transform(X_test.values)
            logger.info("Feature scaling enabled (StandardScaler)")
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

        # ── Evaluate ─────────────────────────────
        metrics = self._evaluate(X_test_scaled, y_test, X_train_scaled, y_train)
        self._log_feature_importance()
        
        # ── Cross-Validation ─────────────────────
        if self.use_cross_validation:
            cv_metrics = self._cross_validate(X_train, y_train)
            logger.info(f"Cross-Validation Results: {cv_metrics}")
            metrics.update({"cv_metrics": cv_metrics})

        return metrics

    def _evaluate(
        self, X_test_scaled, y_test, X_train_scaled, y_train
    ) -> Dict[str, float]:
        """
        Compute and log comprehensive test-set metrics.
        Now includes: Accuracy, AUC-ROC, Precision, Recall, F1, Specificity
        """
        y_pred = self._model.predict(X_test_scaled)
        y_prob = self._model.predict_proba(X_test_scaled)[:, 1]
        y_pred_train = self._model.predict(X_train_scaled)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Specificity (true negative rate)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        train_acc = accuracy_score(y_train, y_pred_train)

        logger.info(
            f"Test Accuracy:  {acc:.4f}  |  Train Accuracy: {train_acc:.4f}  |  AUC-ROC: {auc:.4f}"
        )
        logger.info(f"Precision: {precision:.4f}  |  Recall: {recall:.4f}  |  F1: {f1:.4f}")
        logger.info(f"Specificity: {specificity:.4f}")
        logger.info(
            "\n" + classification_report(y_test, y_pred, target_names=["DOWN", "UP"])
        )
        
        return {
            "accuracy": acc,
            "auc_roc": auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "train_accuracy": train_acc,
        }

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Time Series Cross-Validation (5-fold by default).
        Respects temporal order to avoid look-ahead bias.
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        cv_scores = {"accuracies": [], "aucs": [], "f1_scores": []}

        fold = 1
        for train_idx, test_idx in tscv.split(X):
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

            # Scale fold data
            if self.use_feature_scaling:
                scaler_fold = StandardScaler()
                X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold.values)
                X_test_fold_scaled = scaler_fold.transform(X_test_fold.values)
            else:
                X_train_fold_scaled = X_train_fold.values
                X_test_fold_scaled = X_test_fold.values

            # Train and evaluate fold
            model_fold = lgb.LGBMClassifier(**self._model.get_params())
            model_fold.fit(X_train_fold_scaled, y_train_fold.values, verbose=-1)

            y_pred_fold = model_fold.predict(X_test_fold_scaled)
            y_prob_fold = model_fold.predict_proba(X_test_fold_scaled)[:, 1]

            fold_acc = accuracy_score(y_test_fold, y_pred_fold)
            fold_auc = roc_auc_score(y_test_fold, y_prob_fold)
            fold_f1 = f1_score(y_test_fold, y_pred_fold, zero_division=0)

            cv_scores["accuracies"].append(fold_acc)
            cv_scores["aucs"].append(fold_auc)
            cv_scores["f1_scores"].append(fold_f1)

            logger.debug(f"Fold {fold}: Acc={fold_acc:.4f}, AUC={fold_auc:.4f}, F1={fold_f1:.4f}")
            fold += 1

        # Calculate averages
        cv_metrics = {
            "mean_accuracy": np.mean(cv_scores["accuracies"]),
            "std_accuracy": np.std(cv_scores["accuracies"]),
            "mean_auc": np.mean(cv_scores["aucs"]),
            "std_auc": np.std(cv_scores["aucs"]),
            "mean_f1": np.mean(cv_scores["f1_scores"]),
            "std_f1": np.std(cv_scores["f1_scores"]),
        }

        return cv_metrics

    def _log_feature_importance(self) -> None:
        """Log top-15 feature importances (was top-10)."""
        importances = self._model.feature_importances_
        ranked = sorted(
            zip(self.feature_cols, importances), key=lambda x: -x[1]
        )
        lines = ["Feature Importances (top 15):"]
        for i, (feat, imp) in enumerate(ranked[:15], 1):
            lines.append(f"  {i:2d}. {feat:<25} {imp:.0f}")
        logger.info("\n".join(lines))

    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    def save(self, path: str = None) -> None:
        """Pickle model AND scaler to disk."""
        path = path or self.model_path
        # Save both model and scaler as a tuple
        with open(path, "wb") as f:
            pickle.dump((self._model, self._scaler), f)
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
            # Handle both old (model only) and new (model + scaler) formats
            if isinstance(data, tuple):
                self._model, self._scaler = data
            else:
                self._model = data
                self._scaler = None
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

        # Apply scaling if available
        if self._scaler is not None:
            X_scaled = self._scaler.transform(X.values)
        else:
            X_scaled = X.values

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

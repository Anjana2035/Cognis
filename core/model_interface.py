import logging
import numpy as np

logger = logging.getLogger(__name__)


class ModelInterface:
    """
    Abstraction layer for classification models.
    Supports sklearn-style and simple custom models.
    FIX: Now respects temperature-scaled probabilities when set by Fixer.
    """

    def __init__(self, model, task=None, **kwargs):
        # Accept an optional `task` kwarg for compatibility with callers
        # that pass task through. We store it but do not enforce behavior
        # here — it's for downstream components to use if needed.
        self.model = model
        self.task = task
        self._temperature_probs = None  # set by Fixer._temperature_scaling
        self._temperature = None

    def train(self, X_train, y_train):
        if hasattr(self.model, "fit"):
            self.model.fit(X_train, y_train)
            self._temperature_probs = None  # reset after retraining
            logger.info("Model retrained successfully.")
        else:
            raise NotImplementedError("Model does not support 'fit' method")

    def predict(self, X):
        """
        Returns:
            y_pred: (n,)
            probabilities: (n, num_classes)

        FIX: Uses temperature-scaled probabilities if available.
        """

        # FIX: Use temperature-scaled probs if set
        if self._temperature_probs is not None:
            probabilities = self._temperature_probs
            y_pred = np.argmax(probabilities, axis=1)
            return y_pred, probabilities

        # Case 1: Proper probabilistic model
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)
            y_pred = np.argmax(probabilities, axis=1)
            return y_pred, probabilities

        # Case 2: Only predictions available
        elif hasattr(self.model, "predict"):
            y_pred = self.model.predict(X)

            classes = np.unique(y_pred)
            num_classes = len(classes)

            probabilities = np.zeros((len(y_pred), num_classes))
            for i, label in enumerate(y_pred):
                probabilities[i, int(label)] = 1.0

            return y_pred, probabilities

        else:
            raise NotImplementedError("Model does not support prediction")

    def evaluate(self, X, y_true):
        y_pred, probabilities = self.predict(X)

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "probabilities": probabilities
        }

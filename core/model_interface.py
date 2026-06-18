import logging
import numpy as np

logger = logging.getLogger(__name__)


class ModelInterface:
    """
    Abstraction layer for sklearn-style classification models.
    Supports any model with predict / predict_proba / fit.
    """

    def __init__(self, model, task=None, **kwargs):
        # Accept optional `task` kwarg for compatibility with callers
        self.model = model
        self.task = task
        self._temperature_probs = None
        self._temperature = None

    def train(self, X_train, y_train, **kwargs):
        if hasattr(self.model, "fit"):
            self.model.fit(X_train, y_train, **kwargs)
            self._temperature_probs = None  # reset after retrain
            logger.info("Model retrained successfully.")
        else:
            raise NotImplementedError("Model does not support 'fit'.")

    def predict(self, X):
        """
        Returns:
            y_pred:        (n,)
            probabilities: (n, num_classes)
        """
        # Use temperature-scaled probs if set by Fixer
        if self._temperature_probs is not None:
            probs  = self._temperature_probs
            y_pred = np.argmax(probs, axis=1)
            return y_pred, probs

        if hasattr(self.model, "predict_proba"):
            probs  = self.model.predict_proba(X)
            y_pred = np.argmax(probs, axis=1)
            return y_pred, probs

        if hasattr(self.model, "predict"):
            y_pred  = self.model.predict(X)
            classes = np.unique(y_pred)
            probs   = np.zeros((len(y_pred), len(classes)))
            for i, label in enumerate(y_pred):
                probs[i, int(label)] = 1.0
            return y_pred, probs

        raise NotImplementedError("Model does not support prediction.")

    def evaluate(self, X, y_true):
        y_pred, probabilities = self.predict(X)
        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "probabilities": probabilities
        }

import logging
import numpy as np

logger = logging.getLogger(__name__)


class ModelInterface:

    def __init__(self, model, task=None, **kwargs):
        self._validate_classifier(model)
        self.model = model
        self.task = task
        self._temperature_probs = None
        self._temperature = None

    @staticmethod
    def _validate_classifier(model):
        try:
            import sklearn.base as skbase
            if skbase.is_regressor(model):
                raise ValueError(
                    "Cognis only supports classification models. "
                    f"'{type(model).__name__}' looks like a regressor."
                )
        except ImportError:
            pass  

        if not (hasattr(model, "predict_proba") or hasattr(model, "predict")):
            raise ValueError(
                f"'{type(model).__name__}' exposes no predict/predict_proba — "
                "not a usable classifier."
            )

    def train(self, X_train, y_train, **kwargs):
        if not hasattr(self.model, "fit"):
            raise NotImplementedError("Model does not support 'fit'.")
        self.model.fit(X_train, y_train, **kwargs)
        self._temperature_probs = None  
        self._temperature = None
        logger.info("Model retrained successfully.")

    def predict(self, X):
       
        if self._temperature_probs is not None:
            probs = self._temperature_probs
            y_pred = np.argmax(probs, axis=1)
            return y_pred, probs

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            y_pred = np.argmax(probs, axis=1)
            return y_pred, probs

        if hasattr(self.model, "predict"):
            y_pred = self.model.predict(X)
            classes = np.unique(y_pred).astype(int)
            n_classes = int(classes.max()) + 1 if len(classes) else 1
            probs = np.zeros((len(y_pred), n_classes))
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

import numpy as np


class ModelInterface:
    """
    Abstraction layer for classification models.
    Supports sklearn-style and simple custom models.
    """

    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        if hasattr(self.model, "fit"):
            self.model.fit(X_train, y_train)
        else:
            raise NotImplementedError("Model does not support 'fit' method")

    def predict(self, X):
        """
        Returns:
            y_pred: (n,)
            probabilities: (n, num_classes)
        """

        # Case 1: Proper probabilistic model
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)
            y_pred = np.argmax(probabilities, axis=1)
            return y_pred, probabilities

        # Case 2: Only predictions available
        elif hasattr(self.model, "predict"):
            y_pred = self.model.predict(X)

            # Convert to pseudo-probabilities
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
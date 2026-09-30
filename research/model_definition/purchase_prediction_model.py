from __future__ import annotations

import joblib
import mlflow
import numpy as np


class PurchasePredictionModel(
    mlflow.pyfunc.PythonModel
):

    def load_context(self, context):

        self.prediction_model = joblib.load(
            context.artifacts["prediction_model"]
        )

        self.calibrator = joblib.load(
            context.artifacts["calibrator"]
        )

    def predict(
        self,
        context,
        model_input,
    ):

        raw_probability = (
            self.prediction_model
            .predict_proba(model_input)[:, 1]
        )

        clipped_probability = np.clip(
            raw_probability,
            1e-6,
            1 - 1e-6,
        )

        logit_probability = np.log(
            clipped_probability
            / (1 - clipped_probability)
        )

        return self.calibrator.predict_proba(
            logit_probability.reshape(-1, 1)
        )[:, 1]


mlflow.models.set_model(
    PurchasePredictionModel()
)
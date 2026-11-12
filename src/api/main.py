
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi import HTTPException

from src.preprocessing import WaterDataCleaner, RareCategoryGrouper
import pandas as pd

from src.api.schemas import WaterPointInput, PredictionResponse

# Application configuration
app = FastAPI(
    title="Water Infrastructure Intelligence API",
    description=(
        "API for predicting the operational status "
        "of water infrastructure using machine learning."
    ),
    version="1.0.0"
)


# Model loading
BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "artifacts" / "best_lgbm_pipeline_compatible.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")
print(f"Model type: {type(model).__name__}")
print(f"Model steps: {list(model.named_steps.keys())}")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_water_point(data: WaterPointInput):

    try:
        # 1. Extract the water point ID
        input_data = data.model_dump()
        water_point_id = input_data.pop("id")

        # 2. Convert input data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # 3. Generate prediction
        prediction = model.predict(input_df)[0]

        # 4. Calculate class probabilities
        probabilities = model.predict_proba(input_df)[0]

        # 5. Associate each probability with its class
        classes = model.named_steps["model"].classes_

        probability_dict = {
            str(class_name): float(probability)
            for class_name, probability in zip(classes, probabilities)
        }

        # 6. Return structured response
        return PredictionResponse(
            id=water_point_id,
            prediction=str(prediction),
            probabilities=probability_dict
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while generating the prediction."
        )
      

import os

import requests


# ==========================================
# API CONFIGURATION
# ==========================================

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://water-infrastructure-intelligence.onrender.com"
).rstrip("/")


# ==========================================
# API HEALTH CHECK
# ==========================================

def check_api_health():

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=(10, 120)
        )

        response.raise_for_status()

        return response.json()

    except requests.Timeout as exc:
        raise RuntimeError(
            "The API request timed out. "
            "The backend may be waking up from inactivity."
        ) from exc

    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Unable to connect to the backend API."
        ) from exc

    except requests.HTTPError as exc:
        raise RuntimeError(
            f"API health check failed: HTTP {response.status_code}"
        ) from exc


# ==========================================
# WATER POINT PREDICTION
# ==========================================

def predict_water_point(payload: dict):

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=(10, 180)
        )

        response.raise_for_status()

        return response.json()

    except requests.Timeout as exc:
        raise RuntimeError(
            "Prediction request timed out. "
            "Please try again."
        ) from exc

    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Unable to reach the prediction service."
        ) from exc

    except requests.HTTPError as exc:
        if response.status_code == 422:
            raise ValueError(
                f"Invalid input data: {response.text}"
            ) from exc

        raise RuntimeError(
            f"Prediction failed: HTTP {response.status_code}"
        ) from exc
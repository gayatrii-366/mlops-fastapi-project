import pickle
import numpy as np
from fastapi import FastAPI, Request, Depends
from app.auth import verify_api_key
from fastapi.responses import JSONResponse
from app.schemas import IrisRequest, IrisResponse
from app.logger import setup_logger

# Setup logger
logger = setup_logger()

# Create FastAPI instance
app = FastAPI(title="Iris ML API")

# Load model safely
try:
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise

@app.get("/")
def home():
    logger.info("Health check endpoint called.")
    return {"message": "ML API is running"}

@app.post("/predict", response_model=IrisResponse)
def predict(data: IrisRequest, api_key: str = Depends(verify_api_key)):
    try:
        logger.info(f"Received input: {data}")

        input_data = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])

        prediction = model.predict(input_data)[0]

        logger.info(f"Prediction result: {prediction}")

        return IrisResponse(prediction=int(prediction))

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong"}
    )
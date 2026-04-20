import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.auth import verify_api_key
from fastapi.responses import JSONResponse, RedirectResponse
from app.schemas import IrisRequest, IrisResponse
from app.logger import setup_logger

# Setup logger
logger = setup_logger()

# Create FastAPI instance
app = FastAPI(
    title="Iris ML API",
    description="Iris flower classifier powered by scikit-learn, served via FastAPI.",
    version="1.0.0",
)

# Allow the frontend (opened as file:// or on a different port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend UI at /ui
_frontend = Path(__file__).parent.parent / "frontend"
if _frontend.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend), html=True), name="frontend")

# Load model safely
try:
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise

@app.get("/", include_in_schema=False)
def home():
    logger.info("Health check endpoint called.")
    return {"message": "ML API is running", "frontend": "/ui", "docs": "/docs"}

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
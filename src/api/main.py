from fastapi import FastAPI
from .routes import router
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn using a machine learning model",
    version="1.0.0",
)

# Add GZip middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include the router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
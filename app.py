from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
from io import BytesIO
import numpy as np

app = FastAPI(title="Laptop Price Prediction API")

try:
    model = joblib.load('laptop_price_model.pkl')
    print("Модель успешно загружена")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")


@app.get("/")
async def root():
    return {"message": "Laptop Price Prediction API"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))

        predictions = model.predict(df)

        return JSONResponse({
            "predictions": predictions.tolist(),
            "status": "success",
            "num_predictions": len(predictions)
        })

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "status": "error"}
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
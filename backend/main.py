from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import train_model, predict, get_all_models
import pandas as pd
import io

app = FastAPI(title="MLOps Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "MLOps Platform is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "message": "Dataset uploaded successfully!"
    }

@app.post("/train")
async def train(file: UploadFile = File(...), target_column: str = "target"):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    result = train_model(df, target_column)
    return {"status": "success", "result": result}

@app.post("/predict")
async def make_prediction(input_data: dict):
    result = predict(input_data)
    return result

@app.get("/models")
def list_models():
    return {"models": get_all_models()}
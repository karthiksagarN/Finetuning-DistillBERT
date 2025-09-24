from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib

# Initialize FastAPI
app = FastAPI(title="DistilBERT Inference API")

# Load model and tokenizer
model_path = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
le = joblib.load("./saved_model/label_encoder.pkl")

# Define request body
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    # Tokenize input
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    return {"text": input.text, "predicted_class": le.inverse_transform([predicted_class_id])[0]}

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize FastAPI
app = FastAPI(title="DistilBERT Inference API")

# Load model and tokenizer
model_path = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define request body
class TextInput(BaseModel):
    text: str

id2label = {
    0: "Bills & Utilities",
    1: "Education",
    2: "Entertainment",
    3: "Food & Drinks",
    4: "Groceries",
    5: "Health & Fitness",
    6: "Income",
    7: "Investments",
    8: "Miscellaneous",
    9: "Shopping",
    10: "Travel & Transport"
}

@app.post("/predict")
def predict(input: TextInput):
    # Tokenize input
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    return {"text": input.text, "predicted_category": id2label[predicted_class_id]}

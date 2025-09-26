from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/Users/karthiksagar/Finetuning-DistillBERT/saved_model",
    repo_id="karthiksagarn/DistillBERT-Financial-Categorizer",
    repo_type="model",
)

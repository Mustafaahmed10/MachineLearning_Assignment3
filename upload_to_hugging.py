from huggingface_hub import login, create_repo, upload_file
import os

# Login to Hugging Face
login(token="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")  # Replace with your HF token

# Repo details
repo_name = "california-housing-regressor"
repo_id = f"i222301ahmedmustafa/{repo_name}"

# Create repo if not exists
create_repo(repo_name, exist_ok=True)

# List of model variants
models = ["bgd_model.pkl", "sgd_model.pkl", "mbgd_model.pkl"]
scaler_path = "scaler.pkl"

# Upload models
for model_file in models:
    upload_file(
        path_or_fileobj=model_file,
        path_in_repo=model_file,
        repo_id=repo_id,
        repo_type="model"
    )

# Upload scaler
upload_file(
    path_or_fileobj=scaler_path,
    path_in_repo=scaler_path,
    repo_id=repo_id,
    repo_type="model"
)

print("All model variants and scaler uploaded to Hugging Face.")

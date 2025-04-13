import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# Choose variant (can be "bgd_model", "sgd_model", or "mbgd_model")
variant = "sgd_model"  # example

# Load model and scaler
repo_id = "i222301ahmedmustafa/california-housing-regressor"
model = joblib.load(hf_hub_download(repo_id=repo_id, filename=f"{variant}.pkl"))
scaler = joblib.load(hf_hub_download(repo_id=repo_id, filename="scaler.pkl"))

# Example input
sample_input = np.array([[8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]])
scaled_input = scaler.transform(sample_input)
prediction = model.predict(scaled_input)

print(f"Predicted house price using {variant}: {prediction[0]:.3f}")

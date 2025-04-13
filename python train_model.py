import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import wandb
from concurrent.futures import ThreadPoolExecutor

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Model configurations for each gradient descent variant
models = {
    "bgd_model.pkl": {
        "label": "Batch Gradient Descent",
        "params": {
            "learning_rate": "constant", "eta0": 0.001, "max_iter": 1000, "tol": 1e-3, "batch_size": None
        }
    },
    "sgd_model.pkl": {
        "label": "Stochastic Gradient Descent",
        "params": {
            "learning_rate": "invscaling", "eta0": 0.01, "max_iter": 1, "tol": None, "batch_size": 1
        }
    },
    "mbgd_model.pkl": {
        "label": "Mini-Batch Gradient Descent",
        "params": {
            "learning_rate": "invscaling", "eta0": 0.01, "max_iter": 20, "tol": None, "batch_size": 64
        }
    }
}

# Function to train and log the model
def train_model(X, y, config_original, save_as, label):
    config = config_original.copy()
    # Initialize W&B
    run = wandb.init(project="gradient-descent-comparison", name=label, config=config)
    batch_size = config.pop("batch_size")

    # Define the metric to compare across models
    wandb.define_metric("epoch")
    wandb.define_metric(f"{label}_rmse", step_metric="epoch")

    if batch_size is None:
        # Full Batch Gradient Descent
        model = SGDRegressor(**config, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        wandb.log({f"{label}_rmse": rmse, "epoch": 1})
    else:
        # SGD / MBGD using partial_fit
        model = SGDRegressor(**config, warm_start=True, random_state=42)
        for epoch in range(config["max_iter"]):
            indices = np.random.permutation(len(X))
            batch_rmse = []
            for i in range(0, len(X), batch_size):
                X_batch = X[indices[i:i+batch_size]]
                y_batch = y[indices[i:i+batch_size]]
                model.partial_fit(X_batch, y_batch)
                y_pred = model.predict(X_batch)
                batch_rmse.append(np.sqrt(mean_squared_error(y_batch, y_pred)))
            wandb.log({f"{label}_rmse": np.mean(batch_rmse), "epoch": epoch + 1})

    # Save the trained model
    joblib.dump(model, save_as)
    wandb.save(save_as)
    wandb.finish()
    print(f"{label} ({save_as}) trained and saved.")

# Function to call for each model
def train_all_models():
    with ThreadPoolExecutor() as executor:
        # Train all models concurrently
        futures = []
        for filename, model_config in models.items():
            futures.append(executor.submit(train_model, X_scaled, y, model_config["params"].copy(), filename, model_config["label"]))
        
        # Wait for all models to complete training
        for future in futures:
            future.result()

# Run the training in parallel
train_all_models()

print("✅ All models trained and logged with W&B.")

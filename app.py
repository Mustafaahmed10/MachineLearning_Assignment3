import joblib
import numpy as np
import wandb
import plotly.graph_objects as go
from flask import Flask, request, render_template
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# A global list to store predictions for demonstration purposes (to show a chart of predictions over time)
predictions_history = []

@app.route("/")
def index():
    return render_template("index.html", predictions_history=predictions_history)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form values
        gd_variant = request.form["gd_variant"]  # e.g., "bgd_model", "sgd_model", "mbgd_model"
        features = [
            float(request.form["MedInc"]),
            float(request.form["HouseAge"]),
            float(request.form["AveRooms"]),
            float(request.form["AveBedrms"]),
            float(request.form["Population"]),
            float(request.form["AveOccup"]),
            float(request.form["Latitude"]),
            float(request.form["Longitude"]),
        ]

        # Load the model and scaler
        repo_id = "i222301ahmedmustafa/california-housing-regressor"
        model = joblib.load(hf_hub_download(repo_id=repo_id, filename=f"{gd_variant}.pkl"))
        scaler = joblib.load(hf_hub_download(repo_id=repo_id, filename="scaler.pkl"))

        input_data = np.array([features])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]

        # Store prediction in history (for charting purposes)
        predictions_history.append(round(prediction, 2))

        # Create a Plotly chart of the predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(predictions_history))), y=predictions_history, mode='lines+markers', name='Predictions'))
        fig.update_layout(title="Predictions Over Time",
                          xaxis_title="Prediction Index",
                          yaxis_title="Predicted House Value (in $100,000s)",
                          showlegend=True)

        # Convert Plotly figure to HTML to render in the page
        chart_html = fig.to_html(full_html=False)

        return render_template("index.html", prediction=round(prediction, 2), selected_variant=gd_variant, chart_html=chart_html)
    
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}", selected_variant=None)

if __name__ == "__main__":
    app.run(debug=True)

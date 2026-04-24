"""
Then open http://127.0.0.1:5000
"""

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "finaltrial.pkl")

artifact     = None
models       = None
preprocessor = None
all_numeric  = None
categorical  = None
target_cols  = None

def load_model():
    global artifact, models, preprocessor, all_numeric, categorical, target_cols
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file not found at '{MODEL_PATH}'.")
        print("          Place finaltrial.pkl next to app.py and restart.")
        return False

    print(f"[INFO] Loading model from '{MODEL_PATH}'...")
    artifact     = joblib.load(MODEL_PATH)
    models       = artifact["models"]
    preprocessor = artifact["preprocessor"]
    all_numeric  = artifact["all_numeric_features"]
    categorical  = artifact["categorical_features"]
    target_cols  = artifact["target_columns"]
    print(f"[INFO] Model loaded. Targets: {target_cols}")
    return True


MODEL_LOADED = load_model()

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    df = df.copy()
    df["Time_x_Temp"]       = df["Time,min  "]       * df["Temperature, C"]
    df["Leach_Conc_x_Time"] = df["Concentration, M"] * df["Time,min  "]
    df["Temp_Squared"]       = df["Temperature, C"]  ** 2
    df["Time_Squared"]       = df["Time,min  "]      ** 2
    df["Acid_to_Reducer"]    = df["Concentration, M"] / (df["Concentration %"] + eps)
    return df


def prepare_input(data: dict) -> pd.DataFrame:
    """Convert the JSON payload from the frontend into the model's expected DataFrame."""
    row = {
        "Li in feed  %"           : float(data.get("li_feed",  0)),
        "Co in feed %"            : float(data.get("co_feed",  0)),
        "Mn in feed  %"           : float(data.get("mn_feed",  0)),
        "Ni in feed %"            : float(data.get("ni_feed",  0)),
        "Concentration, M"        : float(data.get("leach_conc", 1)),
        "Concentration %"         : float(data.get("reduce_conc", 0)),
        "Time,min  "              : float(data.get("time_min", 60)),
        "Temperature, C"          : float(data.get("temp", 60)),
        "Leaching agent "         : str(data.get("leaching_agent", "")).strip().title(),
        "Type of reducing agent " : str(data.get("reducing_agent", "Unknown")).strip().title(),
    }
    df = pd.DataFrame([row])
    df = add_engineered_features(df)
    return df


def predict_all(input_df: pd.DataFrame) -> dict:
    feature_cols = all_numeric + categorical
    X = input_df[feature_cols]
    X_proc = preprocessor.transform(X)

    results = {}
    for metal in target_cols:
        gbm = models.get(metal)
        if gbm is None:
            results[metal] = None
        else:
            val = float(gbm.predict(X_proc)[0])
            # Clamp to physically valid range
            results[metal] = round(max(0.0, min(100.0, val)), 2)
    return results


@app.route("/")
def index():
    return render_template("index.html", model_loaded=MODEL_LOADED)


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Place finaltrial.pkl next to app.py."}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body received."}), 400

    try:
        input_df = prepare_input(data)
        preds    = predict_all(input_df)
        return jsonify({"predictions": preds, "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/sweep", methods=["POST"])
def sweep():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded."}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body received."}), 400

    try:
        base = {
            "li_feed":        float(data.get("li_feed",  0)),
            "co_feed":        float(data.get("co_feed",  0)),
            "mn_feed":        float(data.get("mn_feed",  0)),
            "ni_feed":        float(data.get("ni_feed",  0)),
            "leaching_agent": str(data.get("leaching_agent", "")).strip().title(),
            "leach_conc":     float(data.get("leach_conc", 1)),
            "reducing_agent": str(data.get("reducing_agent", "Unknown")).strip().title(),
            "reduce_conc":    float(data.get("reduce_conc", 0)),
            "temp":           float(data.get("temp", 60)),
            "time_min":       float(data.get("time_min", 60)),
        }

        def run_sweep(param, values):
            out = []
            for v in values:
                inputs = dict(base)
                inputs[param] = v
                df = prepare_input(inputs)
                out.append(predict_all(df))
            return out

        import numpy as np

        temp_vals    = [round(v, 1) for v in np.arange(20, 101, 10).tolist()]
        lconc_vals   = [round(v, 2) for v in np.arange(0.5, 5.1, 0.5).tolist()]
        time_vals    = [round(v, 0) for v in np.arange(10, 181, 20).tolist()]
        rconc_vals   = [round(v, 1) for v in np.arange(0, 10.1, 1.0).tolist()]

        sweeps = {
            "temperature": {
                "label": "Temperature (°C)", "unit": "°C",
                "values": temp_vals, "current": base["temp"],
                "results": run_sweep("temp", temp_vals),
            },
            "leach_conc": {
                "label": "Leaching concentration (M)", "unit": "M",
                "values": lconc_vals, "current": base["leach_conc"],
                "results": run_sweep("leach_conc", lconc_vals),
            },
            "time": {
                "label": "Reaction time (min)", "unit": "min",
                "values": time_vals, "current": base["time_min"],
                "results": run_sweep("time_min", time_vals),
            },
            "reduce_conc": {
                "label": "Reducing agent concentration (%)", "unit": "%",
                "values": rconc_vals, "current": base["reduce_conc"],
                "results": run_sweep("reduce_conc", rconc_vals),
            },
        }

        return jsonify({"sweeps": sweeps, "status": "ok"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"model_loaded": MODEL_LOADED, "status": "running"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
from joblib import load
from flask import Flask, request, url_for, redirect, render_template, jsonify
import numpy as np
import pandas as pd
import json
lin_reg = load('models/lin_reg.joblib')
knn = load('models/knn.joblib')

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("./home.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data["features"]

    # data preparing
    features_array = np.array(features)
    pandas_features = pd.DataFrame([features_array])
    print(features)

    # predict
    pred_reg_lin = lin_reg.predict(pandas_features)
    print(pred_reg_lin)
    prediction_reg_lin = str(pred_reg_lin[0])
    pred_knn = knn.predict(pandas_features)
    print(pred_knn)
    prediction_knn = str(pred_knn[0])
    return {"linReg": prediction_reg_lin, "knn": prediction_knn}


@app.route('/model_health', methods=['GET'])
def model_health():
    models_metrics = {}
    with open('metrics/model_metrics_lin_reg.json') as f:
        lin_reg_metrics = json.load(f)
        models_metrics["lin_reg"] = lin_reg_metrics
    with open('metrics/model_metrics_knn.json') as f:
        knn_metrics = json.load(f)
        models_metrics["knn"] = knn_metrics
    return models_metrics


if __name__ == "__main__":
    app.run()

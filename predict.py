#!/usr/bin/env python
# coding: utf-8

import pickle

import numpy as np
from flask import Flask, jsonify, request

model_file = "model.bin"

with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("houseprice")


@app.route("/predict", methods=["POST"])
def predict():
    house = request.get_json()

    X = dv.transform([house])
    y_pred = model.predict(X)[0]

    result = {
        "houseprice": float(np.expm1(y_pred)),
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)

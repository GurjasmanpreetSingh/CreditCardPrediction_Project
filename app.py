# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# # Load trained model and scaler correctly
# with open("fraud_model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)  # Ensure scaler is a StandardScaler instance

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get input values
#         time = request.form.get("time")
#         amount = request.form.get("amount")

#         # Validate input
#         if not time or not amount:
#             return render_template("index.html", prediction="Error: Please enter both Time and Amount.")

#         time = float(time)
#         amount = float(amount)

#         # Ensure input is reshaped for transformation
#         scaled_data = scaler.transform(np.array([[time, amount]]))  # Fix: Ensure array format

#         # Make prediction
#         prediction = model.predict(scaled_data)
#         result = "Fraudulent Transaction ❌" if prediction[0] == 1 else "Legitimate Transaction ✅"

#         return render_template("index.html", prediction=result)

#     except Exception as e:
#         return render_template("index.html", prediction=f"Error: {str(e)}")

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
import os

# Load trained model and scaler
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values
        time = request.form.get("time")
        amount = request.form.get("amount")

        # Validate input
        if not time or not amount:
            return render_template("index.html", prediction="Error: Please enter both Time and Amount.")

        time = float(time)
        amount = float(amount)

        # Scale input
        scaled_data = scaler.transform(np.array([[time, amount]]))

        # Make prediction
        prediction = model.predict(scaled_data)
        result = "Fraudulent Transaction ❌" if prediction[0] == 1 else "Legitimate Transaction ✅"

        return render_template("index.html", prediction=result, show_graph=True)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

# Route to serve fraud_graph.png dynamically
@app.route("/fraud_graph")
def fraud_graph():
    return send_file("fraud_graph.png", mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)


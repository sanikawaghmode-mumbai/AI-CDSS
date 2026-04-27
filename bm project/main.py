from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"])
        ]

        final_input = np.array([input_data])
        prediction = model.predict(final_input)

        if prediction[0] == 1:
            result = "Diabetic (Consult a doctor)"
        else:
            result = "Not Diabetic"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text="Error in input")

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
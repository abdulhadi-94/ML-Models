import joblib
import pandas as pd

# Load saved model and preprocessing objects
model = joblib.load("Heart Disease Model/LR_Heart_disease.pkl")
scaler = joblib.load("Heart Disease Model/scaler_Heart.pkl")
columns = joblib.load("Heart Disease Model/columns_Heart.pkl")

# Sample patient record (replace values if required)
sample = pd.DataFrame({
    "Age": [62],
    "Sex": ["F"],
    "ChestPainType": ["ATA"],
    "RestingBP": [120],
    "Cholesterol": [240],
    "FastingBS": [0],
    "RestingECG": ["Normal"],
    "MaxHR": [150],
    "ExerciseAngina": ["N"],
    "Oldpeak": [1.0],
    "ST_Slope": ["Up"]
})

# One-Hot Encoding
sample = pd.get_dummies(sample)

# Match training columns
sample = sample.reindex(columns=columns, fill_value=0)

# Scale input
numeric_cols = ['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']

sample[numeric_cols] = scaler.transform(sample[numeric_cols])

# Predict
prediction = model.predict(sample)

# Print result
if prediction[0] == 1:
    print("Prediction: Heart Disease Detected")
else:
    print("Prediction: No Heart Disease")
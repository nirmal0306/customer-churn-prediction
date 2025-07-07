# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Load model and scaler
# model = joblib.load('model_rf.pkl')
# scaler = joblib.load('scaler.pkl')

# st.title("🔁 Customer Churn Prediction App")

# # Input features
# gender = st.selectbox("Gender", ["Male", "Female"])
# SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
# tenure = st.slider("Tenure (in months)", 0, 72)
# MonthlyCharges = st.slider("Monthly Charges", 0.0, 150.0)
# TotalCharges = st.number_input("Total Charges", 0.0, 10000.0)

# # Convert categorical
# gender_male = 1 if gender == "Male" else 0

# # Input vector
# input_data = np.array([[SeniorCitizen, tenure, MonthlyCharges, TotalCharges, gender_male]])
# input_data_scaled = scaler.transform(input_data)

# # Predict
# if st.button("Predict Churn"):
#     prediction = model.predict(input_data_scaled)
#     result = "Yes ❌ (High Risk)" if prediction[0] == 1 else "No ✅ (Safe)"
#     st.subheader(f"Churn Prediction: **{result}**")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")


# Load model and scaler
model = joblib.load('model_rf.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🔁 Customer Churn Prediction App")

# Build user input fields for full required columns
def user_input_features():
    data = {
        'SeniorCitizen': st.selectbox('Senior Citizen', [0, 1]),
        'tenure': st.slider('Tenure (Months)', 0, 72),
        'MonthlyCharges': st.slider('Monthly Charges', 0, 150),
        'TotalCharges': st.number_input('Total Charges'),

        'gender_Male': 1 if st.selectbox('Gender', ['Male', 'Female']) == 'Male' else 0,
        'Contract_Month-to-month': 1,
        'Contract_One year': 0,
        'Contract_Two year': 0,
        # Add all dummy vars here with proper defaults (0 or user selection)
        # Fill others with default 0
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Make sure the columns match your model's expected input order
# You can optionally load expected column order from a file (joblib.dump(X.columns))

# Fill missing dummy columns if needed
expected_cols = scaler.feature_names_in_
for col in expected_cols:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[expected_cols]

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    result = "Yes ❌ (Churn Risk)" if prediction == 1 else "No ✅ (Safe)"
    st.success(f"Prediction: **{result}**")

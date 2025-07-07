# 🔁 Customer Churn Prediction with Streamlit

This project is a machine learning-based Streamlit web application that predicts whether a telecom customer is likely to churn based on historical data.

---

## 📌 Features

- 📊 Uses the Telco Customer Churn dataset (from Kaggle)
- 🧠 Trained with Random Forest Classifier using scikit-learn
- 🧼 Preprocessed categorical and numerical data with feature scaling
- 🧮 Model exported with `joblib`
- 🌐 Live Streamlit app to enter customer details and predict churn
- 📈 Displays key features influencing churn

---

## 🚀 Live Demo

🔗 [**Click here to view the app**](https://your-streamlit-link.streamlit.app)  
_(You can replace this with your actual Streamlit Cloud link)_

---

## 📁 Project Structure

customer-churn-prediction/
│

├── app.py # Streamlit frontend

├── model_rf.pkl # Trained Random Forest model

├── scaler.pkl # StandardScaler used in training

├── churn_notebook.ipynb # Colab notebook for EDA + model training

├── requirements.txt # List of required Python packages

└── README.md # This file


---

## 🧠 Tech Stack

- Python
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn
- Streamlit
- Google Colab
- Joblib

---

## 📊 Dataset

We used the [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle, which contains customer demographics and service usage data.

---

## ⚙️ How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/nirmal0306/customer-churn-prediction.git
   cd customer-churn-prediction

# Install dependencies:
pip install -r requirements.txt

# Run the Streamlit app:
streamlit run app.py

# Example Inputs (in app)
Gender: Male/Female

Senior Citizen: Yes/No

Tenure: 1 to 72 months

Monthly & Total Charges

Contract type, Internet service, etc.

## 🙋‍♂️ Author

# Nirmal Barot

# Final-year M.Sc. IT student | Passionate about Data Science & Full Stack

# 📧 Connect: LinkedIn : https://www.linkedin.com/in/nirmal-barot-0b4356254

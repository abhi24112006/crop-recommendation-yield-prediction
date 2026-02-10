# 🌾 Crop Advisory System with Yield Prediction (India)
## 📌 Project Overview

Agriculture in India is highly dependent on factors such as rainfall, season, region, and input usage. Choosing the right crop and estimating its expected yield is critical for improving productivity and reducing risk for farmers.

This project presents a machine learning–based crop advisory system that:

**recommends the top suitable crops for given conditions, and

predicts the expected yield for each recommended crop.**

The system is built using real agricultural data from India and deployed with a Streamlit web interface for interactive use.

## 🎯 Objectives

Recommend suitable crops based on regional and environmental conditions

Predict expected crop yield for informed decision-making

Build an end-to-end ML pipeline (EDA → modeling → deployment)

Provide an easy-to-use frontend for non-technical users

## 📊 Dataset Description

The dataset contains agricultural records with the following features:

Column	Description
Crop	Name of the crop
Crop_Year	Year of cultivation
Season	Cropping season (Kharif, Rabi, Whole Year)
State	Indian state
Area	Cultivated area (hectares)
Production	Total production (metric tons)
Annual_Rainfall	Annual rainfall (mm)
Fertilizer	Fertilizer used (kg)
Pesticide	Pesticide used (kg)
Yield	Crop yield (target for regression)

## 🧠 System Architecture

The system is divided into two machine learning models:

1️⃣ Crop Recommendation Model

Type: Multiclass Classification

Algorithm: Random Forest Classifier

Output: Top-3 recommended crops

2️⃣ Yield Prediction Model

Type: Regression

Algorithm: Random Forest Regressor

Output: Expected yield for each recommended crop

Both models are combined into a single advisory system.

## ⚙️ Machine Learning Pipeline

Data Cleaning & Preprocessing

Exploratory Data Analysis (EDA)

Feature Encoding & Scaling

Model Training & Evaluation

Model Serialization (joblib)

End-to-End System Integration

Frontend Deployment using Streamlit

## 📈 Model Performance
Crop Recommendation

Accuracy (Top-1): ~50%

**Top-K recommendation used instead of strict accuracy, as multiple crops can be suitable under the same conditions.**

Yield Prediction (Random Forest)

R² Score: ~0.98

MAE: ~9

Strong agreement between actual and predicted values

## 🖥️ Frontend (Streamlit Web App)

A lightweight web interface was built using Streamlit, allowing users to:

Select State and Season

Enter Area, Rainfall, Fertilizer, and Pesticide usage

Get Top-3 crop recommendations

View expected yield for each crop

**How to Run the App**
python -m streamlit run app.py

## 📂 Project Structure
crop-advisory-system/
│
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── crop_model.pkl
│   └── yield_model.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_crop_recommendation.ipynb
│   ├── 04_yield_prediction.ipynb
│   └── 05_system_demo.ipynb
├── README.md
└── requirements.txt

**🧪 Example Output**
Recommended Crop	Expected Yield
Maize	98.94
Jowar	1.56
Rapeseed & Mustard	0.87
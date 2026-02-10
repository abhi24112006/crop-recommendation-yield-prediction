import streamlit as st
import pandas as pd
import numpy as np
import joblib


# -----------------------------
# Advisory system function
# -----------------------------
def advisory_system(input_df, crop_model, yield_model,
                    crop_preprocessor, yield_preprocessor, k=3):

    X_crop_enc = crop_preprocessor.transform(input_df)
    probs = crop_model.predict_proba(X_crop_enc)[0]

    top_k_idx = np.argsort(probs)[-k:][::-1]
    crops = crop_model.classes_

    recommendations = []

    for idx in top_k_idx:
        crop = crops[idx]

        temp = input_df.copy()
        temp["Crop"] = crop

        X_yield_enc = yield_preprocessor.transform(temp)
        predicted_yield = yield_model.predict(X_yield_enc)[0]

        recommendations.append((crop, predicted_yield))

    return recommendations


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌾 Crop Advisory System")


# -----------------------------
# Load models & preprocessors
# -----------------------------
crop_model = joblib.load("models/crop_model.pkl")
yield_model = joblib.load("models/yield_model.pkl")
crop_preprocessor = joblib.load("data/processed/preprocess_crop.pkl")
yield_preprocessor = joblib.load("data/processed/preprocess_yield.pkl")

st.success("Models loaded successfully!")


# -----------------------------
# Extract dynamic categories
# -----------------------------
cat_features = crop_preprocessor.named_transformers_["cat"]
cat_names = list(cat_features.feature_names_in_)

# States
state_index = cat_names.index("State")
states = sorted(cat_features.categories_[state_index])

# Seasons
season_index = cat_names.index("Season")
seasons = sorted(cat_features.categories_[season_index])


# -----------------------------
# User inputs
# -----------------------------
st.subheader("Enter Farm Details")

state = st.selectbox("State", states)
season = st.selectbox("Season", seasons)

area = st.number_input("Area (hectares)", min_value=0.1, value=1.0)
rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=1000.0)
fertilizer = st.number_input("Fertilizer Used (kg)", min_value=0.0, value=500.0)
pesticide = st.number_input("Pesticide Used (kg)", min_value=0.0, value=20.0)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Get Crop Recommendation 🌱"):
    input_df = pd.DataFrame([{
        "Crop_Year": 2020,
        "Season": season,
        "State": state,
        "Area": area,
        "Annual_Rainfall": rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide
    }])

    results = advisory_system(
        input_df,
        crop_model,
        yield_model,
        crop_preprocessor,
        yield_preprocessor,
        k=3
    )

    output_df = pd.DataFrame(
        results,
        columns=["Recommended Crop", "Expected Yield"]
    )

    st.subheader("Recommended Crops & Expected Yield")
    st.table(output_df)

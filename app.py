import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained XGBoost model using joblib
model = joblib.load("xgb_model_colab.pkl")

# Define the input features
features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "NewBMI_Obesity 1",
    "NewBMI_Obesity 2",
    "NewBMI_Obesity 3",
    "NewBMI_Overweight",
    "NewBMI_Underweight",
    "NewInsulinScore_Normal",
    "NewGlucose_Low",
    "NewGlucose_Normal",
    "NewGlucose_Overweight",
    "NewGlucose_Secret",
]

# Define average values for pre-filling input fields
average_values = {
    "Pregnancies": 3,  # Average number of pregnancies
    "Glucose": 95.0,  # Average fasting glucose level
    "BloodPressure": 120.0,  # Average systolic blood pressure
    "SkinThickness": 17.5,  # Average skin thickness
    "Insulin": 12.5,  # Average fasting insulin level
    "BMI": 22.0,  # Average BMI
    "DiabetesPedigreeFunction": 0.35,  # Average Diabetes Pedigree Function
    "Age": 30,  # Average age at childbirth
}

# Create the input form
st.title("Diabetes Risk Prediction App")


# Use sidebar for input fields to keep the main page clean
with st.sidebar:
    st.header("Enter Your Information")
    
    pregnancies = st.number_input(f"Number of Pregnancies | Average: {average_values['Pregnancies']}", min_value=0, step=1)

    glucose = st.number_input(f"Glucose Level (mg/dL) | Average: {average_values['Glucose']}", min_value=0.0, step=0.1)

    blood_pressure = st.number_input(f"Blood Pressure (mmHg) [Systolic] | Average: {average_values['BloodPressure']}", min_value=0.0, step=0.1)

    skin_thickness = st.number_input(f"Skin Thickness (mm) | Average: {average_values['SkinThickness']}", min_value=0.0, step=0.1)

    insulin = st.number_input(f"Insulin (µU/mL) | Average: {average_values['Insulin']}", min_value=0.0, step=0.1)

    bmi = st.number_input(f"BMI (kg/m^2) | Average: {average_values['BMI']}", min_value=0.0, step=0.1)

    pedigree_function = st.number_input(f"Diabetes Pedigree Function | Average: {average_values['DiabetesPedigreeFunction']}", min_value=0.0, step=0.1)

    age = st.number_input(f"Age (years) | Average: {average_values['Age']}", min_value=0, step=1)

# Create a button to trigger prediction
if st.button("Predict Diabetes Risk"):
    # Create a DataFrame from the input values
    inputs = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree_function,
        "Age": age,
    }
    input_df = pd.DataFrame([inputs])

    # Initialize new features with default values
    input_df["NewBMI_Obesity 1"] = 0
    input_df["NewBMI_Obesity 2"] = 0
    input_df["NewBMI_Obesity 3"] = 0
    input_df["NewBMI_Overweight"] = 0
    input_df["NewBMI_Underweight"] = 0
    input_df["NewInsulinScore_Normal"] = 0
    input_df["NewGlucose_Low"] = 0
    input_df["NewGlucose_Normal"] = 0
    input_df["NewGlucose_Overweight"] = 0
    input_df["NewGlucose_Secret"] = 0

    # Apply BMI categorization
    if input_df["BMI"].values < 18.5:
        input_df["NewBMI_Underweight"] = 1
    elif 18.5 <= input_df["BMI"].values <= 24.9:
        input_df["NewBMI_Obesity 1"] = 1  # Assuming normal weight categorization
    elif 24.9 < input_df["BMI"].values <= 29.9:
        input_df["NewBMI_Overweight"] = 1
    elif 29.9 < input_df["BMI"].values <= 34.9:
        input_df["NewBMI_Obesity 1"] = 1
    elif 34.9 < input_df["BMI"].values <= 39.9:
        input_df["NewBMI_Obesity 2"] = 1
    else:
        input_df["NewBMI_Obesity 3"] = 1

    # Apply Insulin categorization
    if 16 <= input_df["Insulin"].values <= 166:
        input_df["NewInsulinScore_Normal"] = 1

    # Apply Glucose categorization
    if input_df["Glucose"].values <= 70:
        input_df["NewGlucose_Low"] = 1
    elif 70 < input_df["Glucose"].values <= 99:
        input_df["NewGlucose_Normal"] = 1
    elif 99 < input_df["Glucose"].values <= 126:
        input_df["NewGlucose_Overweight"] = 1
    else:
        input_df["NewGlucose_Secret"] = 1

    # Make the prediction
    prediction = model.predict(input_df[features])
    prediction_proba = model.predict_proba(input_df[features])

    # Display the prediction result
    if prediction == 1:
        st.error("The prediction indicates the patient has a high risk of diabetes.")
        st.write(f"Probability of having diabetes: {prediction_proba[0][1]:.2%}")
        st.write("**Recommendations:**")
        st.write("- Consider consulting with a healthcare provider.")
        st.write("- Maintain a balanced diet low in sugars and refined carbs.")
        st.write("- Increase physical activity to at least 150 minutes per week.")
        st.write("- Monitor your blood sugar levels regularly.")
    else:
        st.success("The prediction indicates the patient has a low risk of diabetes.")
        st.write(f"Probability of not having diabetes: {prediction_proba[0][0]:.2%}")
        st.write("**Keep up the good work** Continue to maintain a healthy lifestyle.")

    # Data Analysis Section
    st.header("Data Analysis")

    # 1. Feature Importance
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"feature": features, "importance": feature_importance}
    )
    feature_importance_df = feature_importance_df.sort_values(
        "importance", ascending=False
    )

    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=feature_importance_df.head(10), ax=ax)
    ax.set_title("Top 10 Most Important Features")
    st.pyplot(fig)

    st.subheader("Your Data vs. Average")
    comparison_df = pd.DataFrame(
        {
            "Feature": average_values.keys(),
            "Your Value": [inputs[key] for key in average_values.keys()],
            "Average Value": average_values.values(),
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.plot(
        x="Feature", y=["Your Value", "Average Value"], kind="bar", ax=ax
    )
    plt.title("Your Data vs. Average")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 3. Risk Factors Analysis
    st.subheader("Risk Factors Analysis")
    risk_factors = []
    if inputs["BMI"] > 25:
        risk_factors.append("High BMI")
    if inputs["Glucose"] > 100:
        risk_factors.append("High Glucose")
    if inputs["BloodPressure"] > 130:
        risk_factors.append("High Blood Pressure")
    if inputs["Age"] > 40:
        risk_factors.append("Age")

    if risk_factors:
        st.write(
            "Based on your input, the following factors may contribute to increased diabetes risk:"
        )
        for factor in risk_factors:
            st.write(f"- {factor}")
    else:
        st.write(
            "No significant risk factors identified based on the provided information."
        )

st.write(
    "**Please note:** This app is for informational purposes only. Consult a healthcare professional for personalized advice."
)

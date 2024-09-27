import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained XGBoost model using joblib
model = joblib.load('xgb_model_colab.pkl')

# Define the input features
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
            'DiabetesPedigreeFunction', 'Age', 'NewBMI_Obesity 1', 'NewBMI_Obesity 2', 
            'NewBMI_Obesity 3', 'NewBMI_Overweight', 'NewBMI_Underweight', 
            'NewInsulinScore_Normal', 'NewGlucose_Low', 'NewGlucose_Normal', 
            'NewGlucose_Overweight', 'NewGlucose_Secret']

# Create the input form
st.title("Diabetes Prediction App")


# Create input fields for each feature with dummy values
inputs = {}
for feature in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']:
    if feature in ['Pregnancies', 'Age']:
        inputs[feature] = st.number_input(f"{feature}", min_value=0.0, step=1.0)
    else:
        inputs[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.1)

# Create a button to trigger prediction
if st.button("Predict"):
    # Create a DataFrame from the input values
    input_df = pd.DataFrame([inputs])

    # Initialize new features with default values
    input_df['NewBMI_Obesity 1'] = 0
    input_df['NewBMI_Obesity 2'] = 0
    input_df['NewBMI_Obesity 3'] = 0
    input_df['NewBMI_Overweight'] = 0
    input_df['NewBMI_Underweight'] = 0
    input_df['NewInsulinScore_Normal'] = 0
    input_df['NewGlucose_Low'] = 0
    input_df['NewGlucose_Normal'] = 0
    input_df['NewGlucose_Overweight'] = 0
    input_df['NewGlucose_Secret'] = 0

    # Apply BMI categorization
    if input_df['BMI'].values[0] < 18.5:
        input_df['NewBMI_Underweight'] = 1
    elif 18.5 <= input_df['BMI'].values[0] <= 24.9:
        input_df['NewBMI_Obesity 1'] = 1  # Assuming normal weight categorization
    elif 24.9 < input_df['BMI'].values[0] <= 29.9:
        input_df['NewBMI_Overweight'] = 1
    elif 29.9 < input_df['BMI'].values[0] <= 34.9:
        input_df['NewBMI_Obesity 1'] = 1
    elif 34.9 < input_df['BMI'].values[0] <= 39.9:
        input_df['NewBMI_Obesity 2'] = 1
    else:
        input_df['NewBMI_Obesity 3'] = 1

    # Apply Insulin categorization
    if 16 <= input_df['Insulin'].values[0] <= 166:
        input_df['NewInsulinScore_Normal'] = 1

    # Apply Glucose categorization
    if input_df['Glucose'].values[0] <= 70:
        input_df['NewGlucose_Low'] = 1
    elif 70 < input_df['Glucose'].values[0] <= 99:
        input_df['NewGlucose_Normal'] = 1
    elif 99 < input_df['Glucose'].values[0] <= 126:
        input_df['NewGlucose_Overweight'] = 1
    else:
        input_df['NewGlucose_Secret'] = 1

    # Log input DataFrame for debugging
    st.write("Input DataFrame for prediction:")
    st.dataframe(input_df)

    # Make the prediction
    prediction = model.predict(input_df[features])[0]

    # Log the raw prediction value
    st.write(f"Raw prediction output: {prediction}")

    # Display the prediction result
    if prediction == 1:
        st.error("The prediction indicates a high risk of diabetes.")
        st.write("**Recommendations:**")
        st.write("- Consider consulting with a healthcare provider.")
        st.write("- Maintain a balanced diet low in sugars and refined carbs.")
        st.write("- Increase physical activity to at least 150 minutes per week.")
        st.write("- Monitor your blood sugar levels regularly.")
    else:
        st.success("The prediction indicates a low risk of diabetes.")
        st.write("**Keep up the good work!** Continue to maintain a healthy lifestyle.")


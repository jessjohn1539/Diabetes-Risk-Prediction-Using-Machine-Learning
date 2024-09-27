# Diabetes Prediction Using Machine Learning

## Problem Statement

Diabetes is a group of metabolic disorders characterized by prolonged high blood sugar levels. If left untreated, diabetes can lead to acute complications such as diabetic ketoacidosis, hyperosmolar hyperglycemic state, and even death. Long-term complications include cardiovascular disease, chronic kidney disease, stroke, foot ulcers, and damage to the eyes. This project aims to build a machine learning model that can predict whether a patient has diabetes based on specific diagnostic features.

## Objectives

- Develop a machine learning model using medical diagnostic measurements to predict diabetes.
- Optimize the prediction model using hyperparameter tuning for XGBoost to achieve the best performance.
- Deploy the model in a Streamlit app to allow real-time user interaction and visualization of predictions.
- Provide a user-friendly interface where users can input medical features and receive a prediction along with healthcare recommendations.

## Dataset Description

The dataset used in this project comes from the National Institute of Diabetes and Digestive and Kidney Diseases. It consists of medical predictor variables to diagnostically predict whether a patient has diabetes. The patients in the dataset are all females aged 21 or older of Pima Indian heritage.

- **Pregnancies**: Number of times the patient has been pregnant.
- **Glucose**: Plasma glucose concentration measured after 2 hours in an oral glucose tolerance test.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skin fold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg/(height in m)^2).
- **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history.
- **Age**: The age of the patient in years.
- **Outcome**: The class variable, where 0 represents a negative test for diabetes and 1 represents a positive test for diabetes.

## Model Performance

- The XGBoost model was optimized using hyperparameter tuning and achieved a Cross Validation Score of 0.90.

## How to Run the Application

1. Clone the repository:
    ```bash
    git clone https://github.com/jessjohn1539/Diabetes-Risk-Prediction-Using-Machine-Learning.git .
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```



## Conclusion

This project demonstrates how machine learning can be used to predict diabetes based on medical data. With a high-performing XGBoost model and an interactive Streamlit app, users can easily input their medical information and receive a prediction along with actionable recommendations.


import streamlit as st
import pandas as pd
import joblib

# --- PAGE CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Heart Health Predictor", page_icon="❤️", layout="centered")

# --- 1. LOAD THE PRE-TRAINED MODEL AND OBJECTS ---
# The model and associated objects are loaded once when the app starts.
@st.cache_resource
def load_model():
    model_data = joblib.load('heart_model.joblib')
    return model_data

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
model_columns = model_data['columns']

# --- 2. DEFINE THE USER INTERFACE (UI) ---

# --- HEADER SECTION ---
st.title("Heart Health Prediction ❤️")
st.markdown("This app uses a machine learning model to predict the likelihood of heart disease based on patient data. Please enter the required information below.")
st.markdown("---")

# --- INPUT SECTION ---
st.header("Enter Patient Details")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "Age (years)", 
        min_value=20, max_value=80, value=50,
        help="Enter the patient's age."
    )
    
    chest_pain_type = st.selectbox(
        "Chest Pain Type",
        options=['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'],
        help="Refers to the type of chest pain experienced by the patient."
    )

    max_heart_rate = st.slider(
        "Maximum Heart Rate Achieved",
        min_value=70, max_value=220, value=150,
        help="The highest heart rate measured during a stress test."
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        options=['No', 'Yes'],
        help="Does the patient experience chest pain during exercise?"
    )

with col2:
    ca = st.slider(
        "Number of Major Vessels Blocked",
        min_value=0, max_value=4, value=0,
        help="The number of major blood vessels (0-4) colored by flouroscopy."
    )

    thal = st.selectbox(
        "Thallium Stress Test Result",
        options=['Normal', 'Fixed Defect', 'Reversible Defect', 'Null'],
        help="Result of the thallium heart scan."
    )
    
    oldpeak = st.slider(
        "ST Depression Induced by Exercise",
        min_value=0.0, max_value=6.2, value=1.0, step=0.1,
        help="Measures the depression of the ST segment during exercise relative to rest."
    )

    slope = st.selectbox(
        "Slope of the Peak Exercise ST Segment",
        options=['Upsloping', 'Flat', 'Downsloping'],
        help="The slope of the ST segment during peak exercise."
    )

# --- 3. PREDICTION LOGIC & DISPLAY ---
if st.button("✨ Predict Heart Health", key="predict_button"):
    
    # Create a dictionary from user inputs
    input_data = {
        'age': age,
        'chest_pain_type': chest_pain_type,
        'max_heart_rate': max_heart_rate,
        'exercise_angina': exercise_angina,
        'ca': ca,
        'thal': thal,
        'oldpeak': oldpeak,
        'slope': slope
    }

    # Convert to DataFrame, one-hot encode, and align columns
    input_df = pd.DataFrame([input_data])
    input_df_encoded = pd.get_dummies(input_df)
    input_df_aligned = input_df_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Scale the data
    input_scaled = scaler.transform(input_df_aligned)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    st.markdown("---")
    st.header("Prediction Result")
    
    if prediction[0] == 1:
        st.error("Warning: This person is **LIKELY** to have heart disease.", icon="💔")
        probability = prediction_proba[0][1] * 100
        st.metric(label="Confidence", value=f"{probability:.2f}%")
        st.write("Based on the data provided, the model indicates a high probability of heart disease. It is strongly recommended to consult a medical professional for a comprehensive evaluation.")
    else:
        st.success("Good News: This person is **UNLIKELY** to have heart disease.", icon="✅")
        probability = prediction_proba[0][0] * 100
        st.metric(label="Confidence", value=f"{probability:.2f}%")
        st.write("Based on the data provided, the model indicates a low probability of heart disease. However, maintaining a healthy lifestyle and regular check-ups are always recommended.")

# --- FOOTER ---
st.markdown("---")
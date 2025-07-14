import streamlit as st 
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

st.title("Heart Disease Predictor")

# Create tabs
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

with tab1:
    st.header("Single Patient Prediction")
    
    # Input fields in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=150, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
        cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0, value=200)
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    
    with col2:
        resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202, value=150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0)
        st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    
    # Convert categorical inputs to numeric (FIXED VERSION)
    def preprocess_inputs():
        sex_num = 0 if sex == "Male" else 1
        
        chest_pain_mapping = {
            "Typical Angina": 0,
            "Atypical Angina": 1, 
            "Non-Anginal Pain": 2,
            "Asymptomatic": 3
        }
        chest_pain_num = chest_pain_mapping[chest_pain]
        
        fasting_bs_num = 1 if fasting_bs == "> 120 mg/dl" else 0
        
        resting_ecg_mapping = {
            "Normal": 0,
            "ST-T Wave Abnormality": 1,
            "Left Ventricular Hypertrophy": 2
        }
        resting_ecg_num = resting_ecg_mapping[resting_ecg]
        
        exercise_angina_num = 1 if exercise_angina == "Yes" else 0
        
        st_slope_mapping = {
            "Upsloping": 0,
            "Flat": 1,
            "Downsloping": 2
        }
        st_slope_num = st_slope_mapping[st_slope]
        
        return [age, sex_num, chest_pain_num, resting_bp, cholesterol, 
                fasting_bs_num, resting_ecg_num, max_hr, exercise_angina_num, 
                oldpeak, st_slope_num]
    
    # Model names and file paths
    model_files = ['tree.pkl', 'LogisticRegression.pkl', 'RandomForest.pkl', 'svm.pkl', 'gridrf.pkl']
    algorithm_names = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Grid Search Random Forest']
    
    def load_and_predict(input_features):
        """Load models and make predictions - FIXED VERSION"""
        predictions = []
        probabilities = []
        
        # Convert to DataFrame for consistency
        feature_names = ['age', 'sex', 'chest_pain', 'resting_bp', 'cholesterol', 
                        'fasting_bs', 'resting_ecg', 'max_hr', 'exercise_angina', 
                        'oldpeak', 'st_slope']
        
        input_df = pd.DataFrame([input_features], columns=feature_names)
        
        for i, model_file in enumerate(model_files):
            try:
                if os.path.exists(model_file):
                    # Load the model
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Make prediction
                    if hasattr(model, 'predict'):
                        # For models trained with your train_models.py (expecting one-hot encoded data)
                        # We need to create dummy columns to match training format
                        try:
                            # Try direct prediction first
                            pred = model.predict(input_df)[0]
                        except:
                            # If that fails, try with one-hot encoding
                            input_encoded = pd.get_dummies(input_df)
                            # Align with model's expected features
                            if hasattr(model, 'feature_names_in_'):
                                input_aligned = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
                            else:
                                input_aligned = input_encoded
                            pred = model.predict(input_aligned)[0]
                        
                        predictions.append(pred)
                        
                        # Get probability if available
                        if hasattr(model, 'predict_proba'):
                            try:
                                prob = model.predict_proba(input_df)[0][1]
                            except:
                                if hasattr(model, 'feature_names_in_'):
                                    input_aligned = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
                                else:
                                    input_aligned = input_encoded
                                prob = model.predict_proba(input_aligned)[0][1]
                            probabilities.append(prob)
                        else:
                            probabilities.append(None)
                    else:
                        st.error(f"Model {model_file} is not properly trained")
                        predictions.append(0)
                        probabilities.append(None)
                else:
                    st.warning(f"Model file {model_file} not found")
                    predictions.append(0)
                    probabilities.append(None)
            except Exception as e:
                st.error(f"Error with {algorithm_names[i]}: {str(e)}")
                predictions.append(0)
                probabilities.append(None)
        
        return predictions, probabilities
    
    # Prediction button
    if st.button("Predict Heart Disease", type="primary"):
        st.subheader("Results")
        st.markdown("-------------------------------")
        
        # Get preprocessed inputs
        input_features = preprocess_inputs()
        
        # Make predictions
        predictions, probabilities = load_and_predict(input_features)
        
        # Display results for each model
        for i in range(len(algorithm_names)):
            if i < len(predictions):
                st.subheader(algorithm_names[i])
                
                col1, col2 = st.columns(2)
                with col1:
                    if predictions[i] == 0:
                        st.success("No heart disease detected")
                    else:
                        st.error("Heart disease detected")
                
                with col2:
                    if probabilities[i] is not None:
                        st.write(f"Confidence: {probabilities[i]:.2%}")
                    else:
                        st.write("Confidence: Not available")
                
                st.markdown("-------------------------------")
        
        # Overall consensus
        if predictions:
            st.subheader("Overall Consensus")
            risk_count = sum(predictions)
            total_models = len([p for p in predictions if p is not None])
            
            if total_models > 0:
                consensus_ratio = risk_count / total_models
                
                if consensus_ratio >= 0.6:
                    st.error(f"HIGH RISK - {risk_count}/{total_models} models predict heart disease")
                elif consensus_ratio >= 0.3:
                    st.warning(f"MODERATE RISK - {risk_count}/{total_models} models predict heart disease")
                else:
                    st.success(f"LOW RISK - {risk_count}/{total_models} models predict heart disease")
            
            st.markdown("-------------------------------")
            st.info("This tool is for educational purposes only. Always consult healthcare professionals for medical decisions.")

with tab2:
    st.header("Bulk Prediction")
    st.info("Upload a CSV file with patient data for bulk predictions")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            bulk_data = pd.read_csv(uploaded_file)
            st.write("Data preview:")
            st.dataframe(bulk_data.head())
            
            if st.button("Run Bulk Prediction"):
                st.info("Bulk prediction feature - implement based on your specific needs")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

with tab3:
    st.header("Model Information")
    
    st.write("Available Models:")
    for i, (model_file, algo_name) in enumerate(zip(model_files, algorithm_names)):
        if os.path.exists(model_file):
            st.success(f"✅ {algo_name} - {model_file}")
        else:
            st.error(f"❌ {algo_name} - {model_file} (Not Found)")
    
    st.markdown("---")
    st.write("""
    **Model Descriptions:**
    - **Decision Tree**: Makes decisions through a series of questions
    - **Logistic Regression**: Uses statistical method for binary classification
    - **Random Forest**: Combines multiple decision trees
    - **Support Vector Machine**: Finds optimal boundary between classes
    - **Grid Search Random Forest**: Optimized Random Forest with best parameters
    """)

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.write("""
    Heart Disease Prediction App using Multiple ML Models
    
    **Requirements:**
    1. Trained model files (.pkl)
    2. Python libraries: streamlit, pandas, numpy, scikit-learn
    """)
    
    st.markdown("### Setup for VS Code")
    st.code("""
# Install required packages
pip install streamlit pandas numpy scikit-learn

# Run the app
streamlit run heart_disease_app.py
    """)
    
    st.warning("⚠️ Educational use only. Consult medical professionals for health decisions.")
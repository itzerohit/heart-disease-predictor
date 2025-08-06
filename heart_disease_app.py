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

# --- Tab 1: Single Patient Prediction ---
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
    
    # Function to preprocess inputs for single prediction
    # This function creates a DataFrame that can be one-hot encoded later
    def preprocess_single_inputs(age, sex, chest_pain, resting_bp, cholesterol, 
                                 fasting_bs, resting_ecg, max_hr, exercise_angina, 
                                 oldpeak, st_slope):
        
        # Create a dictionary for the single input
        input_data = {
            'Age': age,
            'Sex': 'M' if sex == "Male" else 'F', # Match original CSV values
            'ChestPainType': {
                "Typical Angina": "TA", 
                "Atypical Angina": "ATA", 
                "Non-Anginal Pain": "NAP",
                "Asymptomatic": "ASY"
            }[chest_pain], # Match original CSV values
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': 1 if fasting_bs == "> 120 mg/dl" else 0,
            'RestingECG': {
                "Normal": "Normal",
                "ST-T Wave Abnormality": "ST",
                "Left Ventricular Hypertrophy": "LVH"
            }[resting_ecg], # Match original CSV values
            'MaxHR': max_hr,
            'ExerciseAngina': 'Y' if exercise_angina == "Yes" else 'N', # Match original CSV values
            'Oldpeak': oldpeak,
            'ST_Slope': {
                "Upsloping": "Up",
                "Flat": "Flat",
                "Downsloping": "Down"
            }[st_slope] # Match original CSV values
        }
        return pd.DataFrame([input_data])

    # Model names and file paths
    model_files = ['tree.pkl', 'LogisticRegression.pkl', 'RandomForest.pkl', 'svm.pkl', 'gridrf.pkl']
    algorithm_names = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Grid Search Random Forest']
    
    # Define the expected features after one-hot encoding (based on train_models.py)
    # This list must match the columns the models were trained on
    expected_features = [
        'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
        'Sex_M',
        'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
        'RestingECG_Normal', 'RestingECG_ST',
        'ExerciseAngina_Y',
        'ST_Slope_Flat', 'ST_Slope_Up'
    ]

    def load_and_predict_single(input_df_raw):
        """
        Loads models and makes predictions for a single patient.
        Handles one-hot encoding and column alignment.
        """
        predictions = []
        probabilities = []
        
        # Apply one-hot encoding to the raw input DataFrame
        input_encoded = pd.get_dummies(input_df_raw, drop_first=True)
        
        # Align columns with the expected features used during training
        # Fill missing columns with 0 and keep only expected columns
        input_aligned = input_encoded.reindex(columns=expected_features, fill_value=0)
        
        for i, model_file in enumerate(model_files):
            try:
                if os.path.exists(model_file):
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    if hasattr(model, 'predict'):
                        pred = model.predict(input_aligned)[0]
                        predictions.append(pred)
                        
                        if hasattr(model, 'predict_proba'):
                            prob = model.predict_proba(input_aligned)[0][1]
                            probabilities.append(prob)
                        else:
                            probabilities.append(None)
                    else:
                        st.error(f"Model {model_file} is not properly trained or lacks 'predict' method.")
                        predictions.append(0)
                        probabilities.append(None)
                else:
                    st.warning(f"Model file {model_file} not found. Skipping {algorithm_names[i]}.")
                    predictions.append(0)
                    probabilities.append(None)
            except Exception as e:
                st.error(f"Error with {algorithm_names[i]} prediction: {str(e)}")
                predictions.append(0)
                probabilities.append(None)
        
        return predictions, probabilities
    
    # Prediction button for single patient
    if st.button("Predict Heart Disease", type="primary"):
        st.subheader("Results")
        st.markdown("-------------------------------")
        
        # Get preprocessed inputs as a DataFrame
        input_df_raw = preprocess_single_inputs(age, sex, chest_pain, resting_bp, cholesterol, 
                                                fasting_bs, resting_ecg, max_hr, exercise_angina, 
                                                oldpeak, st_slope)
        
        # Make predictions
        predictions, probabilities = load_and_predict_single(input_df_raw)
        
        # Display results for each model
        for i in range(len(algorithm_names)):
            if i < len(predictions): # Ensure index is within bounds
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
            # Filter out None predictions for consensus calculation
            valid_predictions = [p for p in predictions if p is not None]
            risk_count = sum(valid_predictions)
            total_models = len(valid_predictions)
            
            if total_models > 0:
                consensus_ratio = risk_count / total_models
                
                if consensus_ratio >= 0.6:
                    st.error(f"HIGH RISK - {risk_count}/{total_models} models predict heart disease")
                elif consensus_ratio >= 0.3:
                    st.warning(f"MODERATE RISK - {risk_count}/{total_models} models predict heart disease")
                else:
                    st.success(f"LOW RISK - {risk_count}/{total_models} models predict heart disease")
            else:
                st.warning("No valid model predictions to form a consensus.")
            
            st.markdown("-------------------------------")
            st.info("This tool is for educational purposes only. Always consult healthcare professionals for medical decisions.")

# --- Tab 2: Bulk Prediction ---
with tab2:
    st.header("Bulk Prediction")
    st.write("Upload a CSV file with patient data for bulk predictions")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_bulk = pd.read_csv(uploaded_file)
            st.write("Data preview:")
            st.dataframe(df_bulk.head())
            
            # Check if the required columns exist in the uploaded CSV
            # These are the raw columns before one-hot encoding
            required_raw_columns = [
                'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
            ]
            
            if not all(col in df_bulk.columns for col in required_raw_columns):
                st.error(f"The uploaded CSV is missing one or more required columns. Please ensure it contains: {', '.join(required_raw_columns)}")
            else:
                st.success("CSV file successfully uploaded and validated!")
                
                if st.button("Run Bulk Prediction"):
                    
                    st.write("---")
                    st.subheader("Bulk Prediction Results")
                    
                    # Preprocessing the bulk data to match training format
                    df_processed_bulk = df_bulk.copy()
                    
                    # Apply one-hot encoding, similar to train_models.py
                    df_processed_bulk = pd.get_dummies(df_processed_bulk, drop_first=True)
                    
                    # Align columns with the expected features used during training
                    # This step is crucial to ensure all models receive data in the correct format
                    df_processed_bulk = df_processed_bulk.reindex(columns=expected_features, fill_value=0)

                    # Initialize a DataFrame to store results, including original data
                    results_df = df_bulk.copy()
                    
                    # Store individual model predictions and probabilities
                    all_model_preds = []
                    all_model_probs = []

                    # Load models and make predictions for each row in the bulk data
                    for model_file, algo_name in zip(model_files, algorithm_names):
                        current_predictions = []
                        current_probabilities = []
                        try:
                            if os.path.exists(model_file):
                                with open(model_file, 'rb') as file:
                                    model = pickle.load(file)
                                
                                if hasattr(model, 'predict'):
                                    preds = model.predict(df_processed_bulk)
                                    current_predictions.extend(preds)
                                    
                                    if hasattr(model, 'predict_proba'):
                                        probs = model.predict_proba(df_processed_bulk)[:, 1] # Probability of class 1 (Heart Disease)
                                        current_probabilities.extend(probs)
                                    else:
                                        current_probabilities.extend([None] * len(df_processed_bulk))
                                else:
                                    st.warning(f"Model {model_file} does not have a 'predict' method. Skipping.")
                                    current_predictions.extend([0] * len(df_processed_bulk))
                                    current_probabilities.extend([None] * len(df_processed_bulk))
                            else:
                                st.warning(f"Model file {model_file} not found. Skipping {algo_name}.")
                                current_predictions.extend([0] * len(df_processed_bulk))
                                current_probabilities.extend([None] * len(df_processed_bulk))
                        except Exception as e:
                            st.error(f"Error with {algo_name} during bulk prediction: {str(e)}")
                            current_predictions.extend([0] * len(df_processed_bulk))
                            current_probabilities.extend([None] * len(df_processed_bulk))
                        
                        all_model_preds.append(current_predictions)
                        all_model_probs.append(current_probabilities)

                    # Add individual model predictions and probabilities to results_df
                    for i, algo_name in enumerate(algorithm_names):
                        results_df[f'{algo_name} Prediction'] = [("Heart Disease" if p == 1 else "No Heart Disease") for p in all_model_preds[i]]
                        results_df[f'{algo_name} Confidence'] = [f"{p:.2%}" if p is not None else "N/A" for p in all_model_probs[i]]

                    # Calculate overall consensus for each row
                    # Convert to numpy array for easy summing, handling None values
                    preds_np = np.array(all_model_preds)
                    
                    # Sum predictions for each row (axis=0 sums across models for each patient)
                    consensus_score_per_patient = np.sum(preds_np, axis=0)
                    
                    results_df['Overall Consensus Score'] = consensus_score_per_patient
                    results_df['Overall Risk'] = results_df['Overall Consensus Score'].apply(
                        lambda x: "HIGH RISK" if x >= 3 else "LOW RISK" # 3 or more models predicting heart disease
                    )
                    
                    st.dataframe(results_df)
                    st.info("This tool is for educational purposes only. Always consult healthcare professionals for medical decisions.")

        except Exception as e:
            st.error(f"Error reading or processing uploaded file: {str(e)}")

# --- Tab 3: Model Information ---
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
    - **Decision Tree**: Makes decisions through a series of questions, splitting data based on features.
    - **Logistic Regression**: A statistical method used for binary classification, estimating the probability of an event.
    - **Random Forest**: An ensemble method that combines multiple decision trees to improve prediction accuracy and control overfitting.
    - **Support Vector Machine**: A powerful model that finds an optimal hyperplane to separate data points into different classes.
    - **Grid Search Random Forest**: An optimized Random Forest model where hyperparameters (settings) are systematically tuned using Grid Search to find the best performing combination.
    """)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### About")
    st.write("""
    Heart Disease Prediction App using Multiple ML Models.
    This application demonstrates the use of machine learning to assess the risk of heart disease based on various health parameters.
    """)
    
    st.markdown("### Requirements")
    st.write("""
    To run this application locally, you need:
    1.  Pre-trained model files (.pkl) generated from `train_models.py`.
    2.  Python libraries: `streamlit`, `pandas`, `numpy`, `scikit-learn`.
    """)
    
    st.markdown("### Setup for VS Code")
    st.code("""
# Install required packages
pip install streamlit pandas numpy scikit-learn

# Run the app
streamlit run heart_disease_app.py
    """)
    
    st.warning("⚠️ Educational use only. Consult medical professionals for health decisions.")


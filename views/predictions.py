"""
Predictions page for Streamlit app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from modeling import DiabetesReadmissionModel


def show():
    """Display the predictions page."""
    st.title("📈 Predictions")
    st.markdown("---")
    
    # Check if model is trained
    trained_models = []
    for model_name in ['xgboost', 'random_forest', 'logistic']:
        if f'model_{model_name}' in st.session_state:
            trained_models.append(model_name)
    
    if not trained_models:
        st.warning("⚠️ Please train a model first in the 'Modeling' page.")
        return
    
    # Model selection
    st.subheader("1. Select Model")
    
    selected_model_name = st.selectbox(
        "Choose a trained model:",
        [m.replace('_', ' ').title() for m in trained_models]
    )
    
    model_key = selected_model_name.lower().replace(" ", "_")
    model = st.session_state[f'model_{model_key}']
    
    st.success(f"✅ Using {selected_model_name} model")
    
    st.markdown("---")
    
    # Prediction options
    st.subheader("2. Make Predictions")
    
    tab1, tab2 = st.tabs(["Single Patient Prediction", "Batch Predictions"])
    
    with tab1:
        st.markdown("### Enter Patient Information")
        
        # Get feature names from model
        if 'X_train' in st.session_state:
            feature_names = st.session_state['X_train'].columns.tolist()
            
            st.info(f"ℹ️ This model uses {len(feature_names)} features. Enter values for key features below.")
            
            # Create input form for key features
            col1, col2 = st.columns(2)
            
            # Initialize input dictionary
            patient_input = {}
            
            with col1:
                st.markdown("#### Demographics & History")
                
                # Age
                age_midpoint = st.slider("Age:", 0, 100, 50, 5)
                patient_input['age_midpoint'] = age_midpoint
                
                # Time in hospital
                time_in_hospital = st.slider("Time in hospital (days):", 1, 14, 3)
                patient_input['time_in_hospital'] = time_in_hospital
                
                # Number of lab procedures
                num_lab_procedures = st.slider("Number of lab procedures:", 0, 100, 40)
                patient_input['num_lab_procedures'] = num_lab_procedures
                
                # Number of procedures
                num_procedures = st.slider("Number of procedures:", 0, 6, 0)
                patient_input['num_procedures'] = num_procedures
            
            with col2:
                st.markdown("#### Care Utilization")
                
                # Number of medications
                num_medications = st.slider("Number of medications:", 1, 81, 15)
                patient_input['num_medications'] = num_medications
                
                # Number of outpatient visits
                number_outpatient = st.slider("Outpatient visits (past year):", 0, 40, 0)
                patient_input['number_outpatient'] = number_outpatient
                
                # Number of emergency visits
                number_emergency = st.slider("Emergency visits (past year):", 0, 76, 0)
                patient_input['number_emergency'] = number_emergency
                
                # Number of inpatient visits
                number_inpatient = st.slider("Inpatient visits (past year):", 0, 21, 0)
                patient_input['number_inpatient'] = number_inpatient
            
            # Additional categorical features (one-hot encoded)
            with st.expander("🔧 Advanced: Set Categorical Features"):
                st.markdown("""
                These are one-hot encoded categorical features. 
                In a real application, you would select the original categories.
                For now, all unspecified features will default to 0.
                """)
                
                categorical_features = [f for f in feature_names 
                                       if f not in patient_input.keys()]
                
                if categorical_features:
                    st.multiselect(
                        "Select active categorical features:",
                        categorical_features,
                        help="Select which one-hot encoded features should be set to 1"
                    )
            
            st.markdown("---")
            
            if st.button("🎯 Predict Readmission Risk"):
                with st.spinner("Making prediction..."):
                    # Create full feature vector
                    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
                    
                    # Set the values we have
                    for feature, value in patient_input.items():
                        if feature in input_df.columns:
                            input_df[feature] = value
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    prediction_proba = model.predict_proba(input_df)[0]
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        risk_class = "High Risk" if prediction == 1 else "Low Risk"
                        risk_color = "🔴" if prediction == 1 else "🟢"
                        st.metric("Risk Classification", f"{risk_color} {risk_class}")
                    
                    with col2:
                        readmit_prob = prediction_proba[1] * 100
                        st.metric("Readmission Probability", f"{readmit_prob:.2f}%")
                    
                    with col3:
                        confidence = max(prediction_proba) * 100
                        st.metric("Confidence", f"{confidence:.2f}%")
                    
                    # Risk interpretation
                    st.markdown("---")
                    st.markdown("### 💡 Interpretation")
                    
                    if readmit_prob < 20:
                        st.success("""
                        **Low Risk Patient**: This patient has a low probability of 30-day readmission.
                        Standard discharge procedures should be sufficient.
                        """)
                    elif readmit_prob < 50:
                        st.warning("""
                        **Moderate Risk Patient**: This patient has a moderate risk of readmission.
                        Consider enhanced discharge planning and follow-up within 1-2 weeks.
                        """)
                    else:
                        st.error("""
                        **High Risk Patient**: This patient has a high risk of 30-day readmission.
                        Recommend:
                        - Comprehensive discharge planning
                        - Early follow-up appointment (within 7 days)
                        - Medication reconciliation
                        - Patient education on warning signs
                        - Consider home health services
                        """)
                    
                    # Feature contributions (simplified)
                    with st.expander("📊 Top Contributing Factors"):
                        # Get feature importance if available
                        feature_importance = model.get_feature_importance()
                        if feature_importance is not None:
                            # Show top features that are non-zero in input
                            non_zero_features = [f for f in patient_input.keys() 
                                               if patient_input[f] != 0]
                            
                            relevant_importance = feature_importance[
                                feature_importance['feature'].isin(non_zero_features)
                            ].head(10)
                            
                            st.dataframe(relevant_importance, use_container_width=True)
                        else:
                            st.info("Feature importance not available for this model.")
        else:
            st.error("❌ Training data not found. Please prepare data and train a model first.")
    
    with tab2:
        st.markdown("### Upload Data for Batch Predictions")
        
        st.info("""
        Upload a CSV file with patient data in the same format as the training data.
        The file should contain all required features.
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(batch_df):,} records")
                
                # Show preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(batch_df.head(10), use_container_width=True)
                
                if st.button("🚀 Generate Predictions"):
                    with st.spinner("Generating predictions..."):
                        # Check if features match
                        if 'X_train' in st.session_state:
                            required_features = st.session_state['X_train'].columns.tolist()
                            
                            # Check for missing features
                            missing_features = set(required_features) - set(batch_df.columns)
                            
                            if missing_features:
                                st.warning(f"⚠️ Missing features: {', '.join(list(missing_features)[:10])}")
                                st.info("Will use default values (0) for missing features.")
                                
                                # Add missing features with default values
                                for feature in missing_features:
                                    batch_df[feature] = 0
                            
                            # Ensure correct column order
                            batch_df_ordered = batch_df[required_features]
                            
                            # Make predictions
                            predictions = model.predict(batch_df_ordered)
                            predictions_proba = model.predict_proba(batch_df_ordered)
                            
                            # Add predictions to dataframe
                            result_df = batch_df.copy()
                            result_df['predicted_readmission'] = predictions
                            result_df['readmission_probability'] = predictions_proba[:, 1]
                            result_df['risk_level'] = pd.cut(
                                predictions_proba[:, 1],
                                bins=[0, 0.2, 0.5, 1.0],
                                labels=['Low', 'Moderate', 'High']
                            )
                            
                            st.success("✅ Predictions generated!")
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                high_risk_count = (result_df['risk_level'] == 'High').sum()
                                st.metric("High Risk Patients", f"{high_risk_count:,}")
                            
                            with col2:
                                moderate_risk_count = (result_df['risk_level'] == 'Moderate').sum()
                                st.metric("Moderate Risk Patients", f"{moderate_risk_count:,}")
                            
                            with col3:
                                low_risk_count = (result_df['risk_level'] == 'Low').sum()
                                st.metric("Low Risk Patients", f"{low_risk_count:,}")
                            
                            # Display results
                            st.markdown("### 📊 Prediction Results")
                            st.dataframe(
                                result_df[['predicted_readmission', 'readmission_probability', 'risk_level']].head(20),
                                use_container_width=True
                            )
                            
                            # Download results
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Full Results as CSV",
                                data=csv,
                                file_name="readmission_predictions.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error("❌ Model training data not found.")
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    st.markdown("---")
    
    # Additional information
    with st.expander("ℹ️ About the Predictions"):
        st.markdown("""
        ### How the Model Works
        
        The readmission prediction model analyzes multiple factors including:
        
        1. **Patient Demographics**: Age, gender, race
        2. **Clinical Information**: Diagnoses, procedures, lab tests
        3. **Healthcare Utilization**: Previous admissions, ER visits
        4. **Treatment Details**: Medications, time in hospital
        
        ### Limitations
        
        - Predictions are probabilistic, not deterministic
        - Model accuracy depends on data quality and completeness
        - Should be used as a decision support tool, not sole decision maker
        - Regular model retraining recommended with new data
        
        ### Best Practices
        
        - Use predictions to prioritize care coordination resources
        - Combine with clinical judgment
        - Monitor model performance over time
        - Update model periodically with new data
        """)

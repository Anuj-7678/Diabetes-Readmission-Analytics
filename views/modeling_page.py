"""
Modeling page for Streamlit app.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import time

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data_processing import prepare_modeling_data
from modeling import (
    DiabetesReadmissionModel,
    split_data,
    evaluate_model,
    compare_models,
    get_top_predictors
)
from visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_model_comparison
)


def show():
    """Display the modeling page."""
    st.title("🤖 Machine Learning Modeling")
    st.markdown("---")

    # -------------------------------
    # Check cleaned data
    # -------------------------------
    if 'cleaned_data' not in st.session_state or st.session_state['cleaned_data'] is None:
        st.warning("⚠️ Please clean the data first in the 'Data Analysis' page.")
        return

    df = st.session_state['cleaned_data'].copy()

    if 'readmitted_30_days' not in df.columns:
        st.error("❌ Target variable 'readmitted_30_days' not found.")
        return

    # -------------------------------
    # 1. Data Preparation
    # -------------------------------
    st.subheader("1. Data Preparation")

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test set size:", 0.1, 0.4, 0.2, 0.05)
    with col2:
        random_state = st.number_input("Random seed:", 1, 100, 42)

    if st.button("🎲 Prepare Data for Modeling"):
        with st.spinner("Preparing data..."):
            df_encoded, numerical_features, categorical_features = prepare_modeling_data(df)

            if 'readmitted_30_days' not in df_encoded.columns:
                st.error("❌ Target variable missing after encoding.")
                return

            X_train, X_test, y_train, y_test = split_data(
                df_encoded,
                target_col='readmitted_30_days',
                test_size=test_size,
                random_state=random_state
            )

            st.session_state.update({
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "numerical_features": numerical_features,
                "categorical_features": categorical_features
            })

            st.success("✅ Data prepared successfully!")

    if 'X_train' not in st.session_state:
        st.info("ℹ️ Click 'Prepare Data for Modeling' to continue.")
        return

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']

    st.markdown("---")

    # -------------------------------
    # GLOBAL XGBOOST THRESHOLD (ONE PLACE ONLY)
    # -------------------------------
    st.subheader("🎚️ XGBoost Prediction Threshold")

    xgb_threshold = st.slider(
        "Controls ONLY XGBoost predictions (lower = higher recall)",
        min_value=0.05,
        max_value=0.9,
        value=0.3,
        step=0.05
    )

    st.caption(
        "Logistic Regression and Random Forest always use threshold = 0.5"
    )

    st.markdown("---")

    # -------------------------------
    # 2. Model Selection and Training
    # -------------------------------
    st.subheader("2. Model Selection and Training")

    tab1, tab2 = st.tabs(["Train Single Model", "Compare All Models"])

    # ======================================================
    # TAB 1: TRAIN SINGLE MODEL
    # ======================================================
    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            model_type = st.selectbox(
                "Select model:",
                ["XGBoost", "Random Forest", "Logistic Regression"]
            )

        with col2:
            st.markdown("### Model Info")
            if model_type == "XGBoost":
                st.info("🚀 Gradient boosting model (threshold-adjusted).")
            elif model_type == "Random Forest":
                st.info("🌲 Ensemble of decision trees.")
            else:
                st.info("📊 Linear baseline model.")

        if st.button(f"🎯 Train {model_type}"):
            model_key = model_type.lower().replace(" ", "_")

            with st.spinner(f"Training {model_type}..."):
                start_time = time.time()

                model = DiabetesReadmissionModel(model_type=model_key)
                model.train(X_train, y_train)

                # Threshold logic
                threshold = xgb_threshold if model_type == "XGBoost" else 0.5

                metrics = evaluate_model(
                    model,
                    X_test,
                    y_test,
                    threshold=threshold
                )

                st.session_state[f"model_{model_key}"] = model
                st.session_state[f"metrics_{model_key}"] = metrics

                st.success(f"✅ Model trained in {time.time() - start_time:.2f} seconds!")

        model_key = model_type.lower().replace(" ", "_")
        if f"metrics_{model_key}" in st.session_state:
            model = st.session_state[f"model_{model_key}"]
            metrics = st.session_state[f"metrics_{model_key}"]

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1 Score", f"{metrics['f1']:.4f}")
            col5.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_confusion_matrix(metrics['confusion_matrix']),
                    use_container_width=True
                )
            with col2:
                st.plotly_chart(
                    plot_roc_curve(metrics['roc_curve']),
                    use_container_width=True
                )

            st.markdown("### 🎯 Feature Importance")
            top_n = st.slider("Top features:", 10, 50, 20)
            fi = get_top_predictors(model, top_n)
            if fi is not None:
                st.plotly_chart(
                    plot_feature_importance(fi, top_n),
                    use_container_width=True
                )

    # ======================================================
    # TAB 2: COMPARE ALL MODELS
    # ======================================================
    with tab2:
        st.markdown("### Compare Multiple Models")
        st.info(
            "Logistic Regression & Random Forest use threshold = 0.5. "
            "XGBoost uses the global threshold defined above."
        )

        if st.button("🚀 Train and Compare All Models"):
            with st.spinner("Training all models..."):
                comparison_df = compare_models(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    xgb_threshold=xgb_threshold
                )
                st.session_state['comparison_df'] = comparison_df

        if 'comparison_df' in st.session_state:
            comparison_df = st.session_state['comparison_df']

            st.dataframe(
                comparison_df.style.highlight_max(
                    axis=0,
                    subset=comparison_df.columns[1:]
                ),
                use_container_width=True
            )

            st.plotly_chart(
                plot_model_comparison(comparison_df),
                use_container_width=True
            )

            best_model_idx = comparison_df['ROC AUC'].idxmax()
            best_model = comparison_df.loc[best_model_idx, 'Model']
            best_auc = comparison_df.loc[best_model_idx, 'ROC AUC']

            st.success(
                f"🏆 Best performing model: **{best_model}** "
                f"(ROC AUC = {best_auc:.4f})"
            )
     # 3. Model Insights
    st.subheader("3. Model Insights and Interpretation")
    
    with st.expander("💡 Understanding the Metrics"):
        st.markdown("""
        ### Evaluation Metrics Explained
        
        - **Accuracy**: Percentage of correct predictions. Can be misleading with imbalanced data.
        - **Precision**: Of all predicted readmissions, what percentage were actually readmitted?
        - **Recall**: Of all actual readmissions, what percentage did we predict correctly?
        - **F1 Score**: Harmonic mean of precision and recall. Good overall metric.
        - **ROC AUC**: Area under the ROC curve. Measures model's ability to distinguish between classes.
        
        ### Model Characteristics
        
        **Logistic Regression**
        - ✅ Fast training and prediction
        - ✅ Interpretable coefficients
        - ❌ Assumes linear relationships
        - Best for: Quick baseline, interpretability
        
        **Random Forest**
        - ✅ Handles non-linear relationships
        - ✅ Built-in feature importance
        - ✅ Robust to outliers
        - ❌ Can be slow with large datasets
        - Best for: Feature importance analysis
        
        **XGBoost**
        - ✅ Usually best performance
        - ✅ Handles missing values
        - ✅ Built-in regularization
        - ❌ More hyperparameters to tune
        - Best for: Maximum accuracy
        """)
    
    with st.expander("🎯 Feature Importance Interpretation"):
        st.markdown("""
        ### What Does Feature Importance Tell Us?
        
        Feature importance shows which variables have the most influence on predicting readmission.
        High importance doesn't necessarily mean causation, but indicates strong association.
        
        **Common Important Features:**
        - **Number of inpatient visits**: Previous hospitalizations often predict future ones
        - **Number of diagnoses**: More diagnoses may indicate more complex cases
        - **Discharge disposition**: Where patient goes after discharge matters
        - **Time in hospital**: Length of stay correlates with severity
        - **Age**: Older patients may have higher readmission risk
        - **Number of medications**: Indicates treatment complexity
        
        **Clinical Implications:**
        - High-risk patients might benefit from closer follow-up
        - Resources can be allocated more efficiently
        - Intervention programs can target specific risk factors
        """)
    
    # 4. Save Model
    st.markdown("---")
    st.subheader("4. Save Trained Model")
    
    # Check if any model is trained
    trained_models = []
    for model_name in ['xgboost', 'random_forest', 'logistic']:
        if f'model_{model_name}' in st.session_state:
            trained_models.append(model_name)
    
    if trained_models:
        model_to_save = st.selectbox(
            "Select model to save:",
            [m.replace('_', ' ').title() for m in trained_models]
        )
        
        if st.button("💾 Save Model"):
            model_key = model_to_save.lower().replace(" ", "_")
            model = st.session_state[f'model_{model_key}']
            
            # Save model
            save_path = Path(__file__).parent.parent / "models" / f"{model_key}_model.pkl"
            save_path.parent.mkdir(exist_ok=True)
            
            model.save(str(save_path))
            st.success(f"✅ Model saved to: {save_path}")
    else:
        st.info("ℹ️ Train a model first to enable saving.")

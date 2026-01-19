"""uv pip install streamlit
Main Streamlit application for Diabetes Hospital Analytics.
"""
import streamlit as st
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Page configuration
st.set_page_config(
    page_title="Diabetes Hospital Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application."""
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/hospital.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Go to",
            ["🏠 Home", "📊 Data Overview", "🔍 Data Analysis", "🤖 Modeling", "📈 Predictions"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application provides comprehensive analytics for diabetes hospital admissions,
        including data cleaning, exploration, modeling, and predictions for 30-day readmissions.
        """)
        
        st.markdown("---")
        st.markdown("### Tools & Tech")
        st.markdown("""
        - Python 🐍
        - Streamlit 🎈
        - Scikit-learn 🤖
        - XGBoost 🚀
        - Plotly 📊
        """)
    
    # Main content based on page selection
    if "🏠 Home" in page:
        show_home_page()
    elif "📊 Data Overview" in page:
        from views import data_overview
        data_overview.show()
    elif "🔍 Data Analysis" in page:
        from views import data_analysis
        data_analysis.show()
    elif "🤖 Modeling" in page:
        from views import modeling_page
        modeling_page.show()
    elif "📈 Predictions" in page:
        from views import predictions
        predictions.show()


def show_home_page():
    """Display the home page."""
    st.markdown('<div class="main-header">🏥 Diabetes Hospital Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Complete Analytics Pipeline for Diabetic Hospital Admissions</div>', 
                unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    ## Welcome! 👋
    
    This application provides a complete analytics pipeline for analyzing diabetic hospital admissions.
    It covers everything from data ingestion and cleaning to advanced machine learning models
    for predicting 30-day readmissions.
    
    ### 🎯 Key Features
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📊 Data Overview
        - Load and explore raw data
        - Check data quality
        - Identify missing values
        - View summary statistics
        """)
    
    with col2:
        st.markdown("""
        #### 🔍 Data Analysis
        - Interactive visualizations
        - Feature distributions
        - Correlation analysis
        - Readmission patterns
        """)
    
    with col3:
        st.markdown("""
        #### 🤖 Machine Learning
        - Multiple ML models
        - Model comparison
        - Feature importance
        - Performance metrics
        """)
    
    st.markdown("---")
    
    # Project goals
    st.markdown("""
    ### 🎯 Project Goals
    
    The primary goal is to uncover patterns in hospital readmissions, medications, lab results,
    and diagnosis burden to help clinicians and managers make data-driven decisions that improve
    diabetes care and resource utilization.
    
    ### 📈 Key Metrics
    
    - **30-Day Readmission Rate**: Primary outcome variable
    - **Length of Stay**: Important resource utilization metric
    - **Medication Changes**: Impact on patient outcomes
    - **Lab Procedures**: Correlation with readmission risk
    
    ### 🚀 Getting Started
    
    1. **Data Overview**: Start by exploring the dataset and understanding its structure
    2. **Data Analysis**: Dive deep into feature distributions and patterns
    3. **Modeling**: Train and compare different machine learning models
    4. **Predictions**: Use trained models to make predictions on new data
    
    Use the sidebar to navigate between different sections of the application.
    """)
    
    st.markdown("---")
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        #### Data Pipeline
        1. **Data Ingestion**: Load raw diabetes dataset
        2. **Data Cleaning**: Handle missing values, remove duplicates
        3. **Feature Engineering**: Create new features, categorize diagnoses
        4. **Data Preprocessing**: Encode categorical variables, scale features
        5. **Model Training**: Train multiple ML models
        6. **Evaluation**: Compare models using various metrics
        7. **Deployment**: Make predictions on new data
        
        #### Models Used
        - **Logistic Regression**: Baseline linear model
        - **Random Forest**: Ensemble tree-based model
        - **XGBoost**: Gradient boosting model (typically best performance)
        
        #### Evaluation Metrics
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - ROC AUC
        - Confusion Matrix
        """)


if __name__ == "__main__":
    main()

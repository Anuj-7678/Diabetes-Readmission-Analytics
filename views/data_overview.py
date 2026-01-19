"""
Data Overview page for Streamlit app.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data_processing import load_data, check_missing_values, get_data_summary
from visualization import plot_missing_data


def show():
    """Display the data overview page."""
    st.title("📊 Data Overview")
    st.markdown("---")
    
    # File upload or selection
    st.subheader("1. Load Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        data_source = st.radio(
            "Choose data source:",
            ["Upload CSV file", "Use sample data"],
            horizontal=True
        )
    
    df = None
    
    if data_source == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload your diabetes dataset (CSV)", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df):,} rows and {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    else:
        # Try to load sample data
        sample_paths = [
            Path(__file__).parent.parent.parent / "data" / "diabetic_data.csv",
            Path(__file__).parent.parent.parent / "diabetic_data_cleaned.csv",
            Path(__file__).parent.parent.parent.parent / "data" / "diabetic_data.csv",
            Path(__file__).parent.parent.parent.parent / "diabetic_data_cleaned.csv"
        ]
        
        for path in sample_paths:
            if path.exists():
                try:
                    df = load_data(str(path))
                    st.success(f"✅ Loaded sample data: {len(df):,} rows and {len(df.columns)} columns")
                    break
                except Exception as e:
                    continue
        
        if df is None:
            st.warning("⚠️ Sample data not found. Please upload a CSV file.")
            st.info("Expected data location: `data/diabetic_data.csv` or `diabetic_data_cleaned.csv`")
    
    if df is not None:
        # Store in session state
        st.session_state['raw_data'] = df
        
        st.markdown("---")
        
        # 2. Dataset Summary
        st.subheader("2. Dataset Summary")
        
        summary = get_data_summary(df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{summary['total_records']:,}")
        
        with col2:
            st.metric("Total Columns", summary['total_columns'])
        
        with col3:
            if summary['unique_patients']:
                st.metric("Unique Patients", f"{summary['unique_patients']:,}")
            else:
                st.metric("Unique Patients", "N/A")
        
        with col4:
            st.metric("Memory Usage", f"{summary['memory_usage_mb']:.2f} MB")
        
        st.markdown("---")
        
        # 3. Column Information
        st.subheader("3. Column Information")
        
        # Create column info dataframe
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        
        st.dataframe(col_info, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # 4. Missing Values
        st.subheader("4. Missing Values Analysis")
        
        missing_df = check_missing_values(df)
        
        if not missing_df.empty:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(missing_df, use_container_width=True)
            
            with col2:
                fig = plot_missing_data(missing_df)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")
        
        st.markdown("---")
        
        # 5. Data Preview
        st.subheader("5. Data Preview")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            n_rows = st.slider("Number of rows to display:", 5, 100, 10)
            view_type = st.radio("View:", ["Head", "Tail", "Sample"])
        
        with col2:
            if view_type == "Head":
                st.dataframe(df.head(n_rows), use_container_width=True)
            elif view_type == "Tail":
                st.dataframe(df.tail(n_rows), use_container_width=True)
            else:
                st.dataframe(df.sample(min(n_rows, len(df))), use_container_width=True)
        
        st.markdown("---")
        
        # 6. Summary Statistics
        st.subheader("6. Summary Statistics")
        
        tab1, tab2 = st.tabs(["Numerical Features", "Categorical Features"])
        
        with tab1:
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numerical_cols:
                st.dataframe(df[numerical_cols].describe(), use_container_width=True)
            else:
                st.info("No numerical columns found.")
        
        with tab2:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                cat_summary = []
                for col in categorical_cols[:10]:  # Show first 10 categorical columns
                    top_value = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
                    cat_summary.append({
                        'Column': col,
                        'Unique Values': df[col].nunique(),
                        'Most Frequent': str(top_value),
                        'Frequency': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
                    })
                st.dataframe(pd.DataFrame(cat_summary), use_container_width=True)
            else:
                st.info("No categorical columns found.")
        
        st.markdown("---")
        
        # Download processed data info
        st.subheader("7. Export Data Info")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Column Info as CSV"):
                csv = col_info.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="column_info.csv",
                    mime="text/csv"
                )
        
        with col2:
            if not missing_df.empty and st.button("📥 Download Missing Values Report"):
                csv = missing_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="missing_values_report.csv",
                    mime="text/csv"
                )

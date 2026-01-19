"""
Data Analysis page for Streamlit app.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data_processing import (
    clean_data, 
    create_target_variable, 
    categorize_diagnoses
)
from visualization import (
    plot_target_distribution,
    plot_numerical_distributions,
    plot_categorical_distributions,
    plot_age_distribution,
    plot_readmission_by_category,
    plot_correlation_heatmap
)


def show():
    """Display the data analysis page."""
    st.title("🔍 Data Analysis")
    st.markdown("---")
    
    # Check if data is loaded
    if 'raw_data' not in st.session_state or st.session_state['raw_data'] is None:
        st.warning("⚠️ Please load data first in the 'Data Overview' page.")
        return
    
    df = st.session_state['raw_data'].copy()
    
    # Data cleaning options
    st.subheader("1. Data Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Clean Data"):
            with st.spinner("Cleaning data..."):
                df = clean_data(df)
                df = create_target_variable(df)
                df = categorize_diagnoses(df)
                st.session_state['cleaned_data'] = df
                st.success("✅ Data cleaned successfully!")
    
    with col2:
        if 'cleaned_data' in st.session_state:
            st.success(f"✅ Using cleaned data: {len(st.session_state['cleaned_data']):,} rows")
            df = st.session_state['cleaned_data']
        else:
            st.info("ℹ️ Click 'Clean Data' to preprocess the dataset")
    
    st.markdown("---")
    
    # 2. Target Variable Distribution
    st.subheader("2. Target Variable Distribution")
    
    if 'readmitted_30_days' in df.columns:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = plot_target_distribution(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Key Insights")
            readmit_rate = df['readmitted_30_days'].mean() * 100
            st.metric("30-Day Readmission Rate", f"{readmit_rate:.2f}%")
            
            total_readmitted = df['readmitted_30_days'].sum()
            total_not_readmitted = len(df) - total_readmitted
            
            st.write(f"**Readmitted**: {total_readmitted:,}")
            st.write(f"**Not Readmitted**: {total_not_readmitted:,}")
            
            if readmit_rate < 10:
                st.info("This is an imbalanced dataset. Consider using stratified sampling or class weights.")
    else:
        st.info("Target variable 'readmitted_30_days' not found. Please clean the data first.")
    
    st.markdown("---")
    
    # 3. Numerical Features Distribution
    st.subheader("3. Numerical Features Distribution")
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove ID columns
    numerical_cols = [col for col in numerical_cols 
                     if 'id' not in col.lower() and col not in ['encounter_id', 'patient_nbr']]
    
    if numerical_cols:
        selected_numerical = st.multiselect(
            "Select numerical features to visualize:",
            numerical_cols,
            default=numerical_cols[:4] if len(numerical_cols) >= 4 else numerical_cols
        )
        
        if selected_numerical:
            fig = plot_numerical_distributions(df, selected_numerical)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numerical features available.")
    
    st.markdown("---")
    
    # 4. Categorical Features Distribution
    st.subheader("4. Categorical Features Distribution")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove ID and high cardinality columns
    categorical_cols = [col for col in categorical_cols 
                       if col not in ['encounter_id', 'patient_nbr', 'diag_1', 'diag_2', 'diag_3']]
    
    if categorical_cols:
        selected_categorical = st.multiselect(
            "Select categorical features to visualize:",
            categorical_cols,
            default=categorical_cols[:4] if len(categorical_cols) >= 4 else categorical_cols
        )
        
        if selected_categorical:
            fig = plot_categorical_distributions(df, selected_categorical)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical features available.")
    
    st.markdown("---")
    
    # 5. Age Distribution
    st.subheader("5. Age Distribution")
    
    if 'age_group' in df.columns or 'age' in df.columns:
        fig = plot_age_distribution(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Age information not available.")
    
    st.markdown("---")
    
    # 6. Readmission Analysis by Category
    st.subheader("6. Readmission Analysis by Category")
    
    if 'readmitted_30_days' in df.columns:
        category_options = []
        
        # Find categorical columns suitable for analysis
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < 50 and col not in ['encounter_id', 'patient_nbr']:
                category_options.append(col)
        
        if category_options:
            selected_category = st.selectbox(
                "Select category to analyze:",
                category_options,
                index=0 if 'primary_diagnosis_category' not in category_options 
                      else category_options.index('primary_diagnosis_category')
            )
            
            fig = plot_readmission_by_category(df, selected_category)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                with st.expander("📊 Detailed Statistics"):
                    stats = df.groupby(selected_category)['readmitted_30_days'].agg([
                        ('Total Patients', 'count'),
                        ('Readmitted', 'sum'),
                        ('Readmission Rate', lambda x: f"{x.mean()*100:.2f}%")
                    ]).sort_values('Total Patients', ascending=False)
                    st.dataframe(stats, use_container_width=True)
        else:
            st.info("No suitable categorical columns found for analysis.")
    else:
        st.info("Target variable not available. Please clean the data first.")
    
    st.markdown("---")
    
    # 7. Correlation Analysis
    st.subheader("7. Correlation Analysis")
    
    if numerical_cols:
        show_correlation = st.checkbox("Show correlation heatmap")
        
        if show_correlation:
            # Select columns for correlation
            selected_for_corr = st.multiselect(
                "Select features for correlation analysis:",
                numerical_cols,
                default=numerical_cols[:8] if len(numerical_cols) >= 8 else numerical_cols
            )
            
            if len(selected_for_corr) > 1:
                fig = plot_correlation_heatmap(df, selected_for_corr)
                st.plotly_chart(fig, use_container_width=True)
                
                # Find highly correlated pairs
                corr_matrix = df[selected_for_corr].corr()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            high_corr_pairs.append({
                                'Feature 1': corr_matrix.columns[i],
                                'Feature 2': corr_matrix.columns[j],
                                'Correlation': f"{corr_matrix.iloc[i, j]:.3f}"
                            })
                
                if high_corr_pairs:
                    st.warning("⚠️ Highly correlated features detected (|r| > 0.7):")
                    st.dataframe(pd.DataFrame(high_corr_pairs), use_container_width=True)
            else:
                st.info("Please select at least 2 features for correlation analysis.")
    
    st.markdown("---")
    
    # 8. Custom Analysis
    st.subheader("8. Custom Analysis")
    
    with st.expander("🔧 Pivot Table Analysis"):
        if len(categorical_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                index_col = st.selectbox("Select row (index):", categorical_cols)
            
            with col2:
                column_col = st.selectbox("Select column:", 
                                         [col for col in categorical_cols if col != index_col])
            
            value_col = st.selectbox("Select value:", numerical_cols if numerical_cols else df.columns.tolist())
            agg_func = st.selectbox("Aggregation function:", ['mean', 'sum', 'count', 'median'])
            
            if st.button("Generate Pivot Table"):
                pivot = pd.pivot_table(
                    df, 
                    values=value_col, 
                    index=index_col, 
                    columns=column_col,
                    aggfunc=agg_func,
                    fill_value=0
                )
                st.dataframe(pivot, use_container_width=True)
        else:
            st.info("Need at least 2 categorical columns for pivot table analysis.")

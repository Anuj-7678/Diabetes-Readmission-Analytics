"""
Visualization utilities for diabetes analytics.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, List


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_missing_data(missing_df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart of missing data.
    
    Args:
        missing_df: DataFrame with missing value information
        
    Returns:
        Plotly figure
    """
    if missing_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing data found!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    fig = px.bar(
        missing_df,
        x='Column',
        y='Missing %',
        title='Missing Data by Column',
        labels={'Missing %': 'Missing Percentage (%)'},
        color='Missing %',
        color_continuous_scale='Reds'
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def plot_target_distribution(df: pd.DataFrame, target_col: str = 'readmitted_30_days') -> go.Figure:
    """
    Plot the distribution of the target variable.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        
    Returns:
        Plotly figure
    """
    if target_col not in df.columns:
        return None
    
    value_counts = df[target_col].value_counts()
    labels = ['Not Readmitted', 'Readmitted']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=value_counts.values,
        hole=.3,
        marker_colors=['#2ecc71', '#e74c3c']
    )])
    
    fig.update_layout(
        title='30-Day Readmission Distribution',
        annotations=[dict(text='Readmission', x=0.5, y=0.5, font_size=16, showarrow=False)]
    )
    
    return fig


def plot_numerical_distributions(df: pd.DataFrame, numerical_cols: List[str]) -> go.Figure:
    """
    Plot distributions of numerical features.
    
    Args:
        df: Input dataframe
        numerical_cols: List of numerical column names
        
    Returns:
        Plotly figure with subplots
    """
    from plotly.subplots import make_subplots
    
    n_cols = min(len(numerical_cols), 4)
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=numerical_cols
    )
    
    for idx, col in enumerate(numerical_cols):
        row = idx // n_cols + 1
        col_idx = idx % n_cols + 1
        
        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False),
            row=row, col=col_idx
        )
    
    fig.update_layout(height=300*n_rows, title_text="Distribution of Numerical Features")
    return fig


def plot_categorical_distributions(df: pd.DataFrame, categorical_cols: List[str],
                                   max_categories: int = 10) -> go.Figure:
    """
    Plot distributions of categorical features.
    
    Args:
        df: Input dataframe
        categorical_cols: List of categorical column names
        max_categories: Maximum number of categories to show per feature
        
    Returns:
        Plotly figure
    """
    from plotly.subplots import make_subplots
    
    n_cols = 2
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=categorical_cols
    )
    
    for idx, col in enumerate(categorical_cols):
        row = idx // n_cols + 1
        col_idx = idx % n_cols + 1
        
        value_counts = df[col].value_counts().head(max_categories)
        
        fig.add_trace(
            go.Bar(x=value_counts.index, y=value_counts.values, name=col, showlegend=False),
            row=row, col=col_idx
        )
    
    fig.update_layout(height=400*n_rows, title_text="Distribution of Categorical Features")
    fig.update_xaxes(tickangle=-45)
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = None) -> go.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        
    Returns:
        Plotly figure
    """
    if labels is None:
        labels = ['Not Readmitted', 'Readmitted']
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=labels,
        y=labels,
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=500,
        height=500
    )
    
    return fig


def plot_roc_curve(roc_data: dict) -> go.Figure:
    """
    Plot ROC curve.
    
    Args:
        roc_data: Dictionary with 'fpr', 'tpr', 'thresholds'
        
    Returns:
        Plotly figure
    """
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    roc_auc = np.trapz(tpr, fpr)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        line=dict(color='darkorange', width=2)
    ))
    
    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random classifier',
        line=dict(color='navy', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=600,
        showlegend=True
    )
    
    return fig


def plot_feature_importance(feature_importance_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """
    Plot feature importance.
    
    Args:
        feature_importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        
    Returns:
        Plotly figure
    """
    if feature_importance_df is None or feature_importance_df.empty:
        return None
    
    top_features = feature_importance_df.head(top_n)
    
    fig = go.Figure(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Most Important Features',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=max(400, top_n * 25),
        yaxis={'autorange': 'reversed'}
    )
    
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Plot model comparison metrics.
    
    Args:
        comparison_df: DataFrame with model comparison metrics
        
    Returns:
        Plotly figure
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    fig = go.Figure()
    
    for metric in metrics:
        if metric in comparison_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric]
            ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        height=500
    )
    
    return fig


def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Plot age group distribution.
    
    Args:
        df: Input dataframe
        
    Returns:
        Plotly figure
    """
    if 'age_group' not in df.columns:
        return None
    
    age_counts = df['age_group'].value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Bar(x=age_counts.index, y=age_counts.values, marker_color='lightblue')
    ])
    
    fig.update_layout(
        title='Patient Distribution by Age Group',
        xaxis_title='Age Group',
        yaxis_title='Number of Patients',
        height=500
    )
    
    return fig


def plot_readmission_by_category(df: pd.DataFrame, category_col: str,
                                 target_col: str = 'readmitted_30_days') -> go.Figure:
    """
    Plot readmission rates by category.
    
    Args:
        df: Input dataframe
        category_col: Column to group by
        target_col: Target column
        
    Returns:
        Plotly figure
    """
    if category_col not in df.columns or target_col not in df.columns:
        return None
    
    readmission_rates = df.groupby(category_col)[target_col].agg(['mean', 'count'])
    readmission_rates = readmission_rates.sort_values('mean', ascending=False).head(15)
    
    fig = go.Figure(data=[
        go.Bar(
            x=readmission_rates.index,
            y=readmission_rates['mean'] * 100,
            text=[f'{val:.1f}%' for val in readmission_rates['mean'] * 100],
            textposition='auto',
            marker_color='coral'
        )
    ])
    
    fig.update_layout(
        title=f'30-Day Readmission Rate by {category_col.replace("_", " ").title()}',
        xaxis_title=category_col.replace("_", " ").title(),
        yaxis_title='Readmission Rate (%)',
        height=500
    )
    fig.update_xaxes(tickangle=-45)
    
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numerical_cols: List[str]) -> go.Figure:
    """
    Plot correlation heatmap for numerical features.
    
    Args:
        df: Input dataframe
        numerical_cols: List of numerical columns
        
    Returns:
        Plotly figure
    """
    corr_matrix = df[numerical_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        width=800,
        height=800
    )
    
    return fig

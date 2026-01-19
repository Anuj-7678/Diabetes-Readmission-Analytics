# Diabetic Readmission Analytics \& Prediction System

A comprehensive analytics pipeline for diabetic hospital admissions with machine learning models and an interactive Streamlit UI.

## Features

- 📊 **Data Overview**: Explore and understand the diabetes dataset
- 🔍 **Data Analysis**: Interactive visualizations and feature analysis
- 🤖 **Machine Learning**: Train and compare multiple ML models
- 📈 **Predictions**: Make predictions for individual patients or batch processing
- 🎯 **30-Day Readmission Prediction**: Primary focus on predicting patient readmissions

## Tech Stack

- **Python**: Core programming language
- **UV**: Fast Python package manager
- **Streamlit**: Interactive web interface
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine learning models
- **XGBoost**: Gradient boosting
- **Plotly**: Interactive visualizations
- **SHAP**: Model interpretability

## Installation

### Prerequisites

- Python 3.9 or higher
- UV package manager (will be installed automatically if not present)

### Setup

1. Install UV (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

2. Navigate to the project directory:
```bash
cd diabetes-analytics
```

3. Install dependencies:
```bash
uv sync
```

## Usage

### Running the Streamlit App

```bash
uv run streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Project Structure

```
diabetes-analytics/
├── app.py                  # Main Streamlit application
├── src/                    # Source code modules
│   ├── data_processing.py  # Data cleaning and preprocessing
│   ├── modeling.py         # ML models and training
│   └── visualization.py    # Plotting and visualization utilities
├── pages/                  # Streamlit pages
│   ├── data_overview.py    # Data exploration page
│   ├── data_analysis.py    # Analysis and visualizations
│   ├── modeling_page.py    # Model training and evaluation
│   └── predictions.py      # Prediction interface
├── models/                 # Saved models (created automatically)
└── pyproject.toml         # Project dependencies
```

## Features Overview

### 1. Data Overview
- Load and explore raw datasets
- Check data quality and missing values
- View summary statistics
- Export data reports

### 2. Data Analysis
- Clean and preprocess data
- Visualize feature distributions
- Analyze readmission patterns
- Correlation analysis
- Custom pivot tables

### 3. Machine Learning
- Train multiple models:
  - Logistic Regression (baseline)
  - Random Forest (ensemble)
  - XGBoost (gradient boosting)
- Compare model performance
- View feature importance
- Confusion matrices and ROC curves

### 4. Predictions
- Single patient predictions
- Batch predictions from CSV
- Risk stratification (Low/Moderate/High)
- Downloadable results

## Data Requirements

The application expects a diabetes dataset with the following types of features:

- **Demographics**: age, gender, race
- **Clinical**: diagnoses, procedures, medications
- **Utilization**: admission types, length of stay, prior visits
- **Lab**: number of lab procedures
- **Target**: readmitted status (for training)

Sample datasets should be in CSV format and placed in the project root or `data/` directory.

## Model Performance

Typical model performance metrics:
- **Accuracy**: 60-65%
- **ROC AUC**: 0.65-0.70
- **Precision**: 15-25%
- **Recall**: 30-40%

Note: These metrics reflect the challenging nature of readmission prediction with inherent class imbalance.

## Key Insights

The models typically identify these factors as important predictors:
- Number of previous inpatient visits
- Number of diagnoses
- Discharge disposition
- Time in hospital
- Age
- Number of medications
- Primary diagnosis category

## Development

### Adding New Features

1. **Data Processing**: Add functions to `src/data_processing.py`
2. **Models**: Add new models to `src/modeling.py`
3. **Visualizations**: Add plots to `src/visualization.py`
4. **UI**: Create new pages in `pages/` directory

### Running Tests

```bash
uv run pytest tests/
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset: Diabetes 130-US hospitals for years 1999-2008
- Built with modern Python tools and best practices
- Designed for healthcare analytics professionals

## PowerBI


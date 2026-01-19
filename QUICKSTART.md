# Quick Start Guide

## Installation

1. Install UV:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

2. Install dependencies:
```bash
cd diabetes-analytics
uv sync
```

3. Run the app:
```bash
uv run streamlit run app.py
```

## Using the Application

### Step 1: Data Overview
- Upload your diabetes CSV file or use sample data
- Review data quality and missing values
- Understand the dataset structure

### Step 2: Data Analysis
- Click "Clean Data" to preprocess
- Explore feature distributions
- Analyze readmission patterns
- View correlations

### Step 3: Modeling
- Click "Prepare Data for Modeling"
- Choose a model (XGBoost recommended)
- Train the model
- Review performance metrics
- Compare all models if needed

### Step 4: Predictions
- Select a trained model
- Enter patient information for single prediction
- Or upload CSV for batch predictions
- Download results

## Tips

- Use XGBoost for best performance
- Check feature importance to understand predictions
- High risk patients (>50%) need intensive follow-up
- Model works best with complete feature data

## Troubleshooting

**Issue**: Data not loading
- Ensure CSV file is in correct format
- Check file path

**Issue**: Model training fails
- Clean data first in Data Analysis page
- Ensure target variable exists

**Issue**: Predictions not working
- Train a model first
- Ensure all required features are present

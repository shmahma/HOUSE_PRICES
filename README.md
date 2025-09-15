# House Prices Prediction

## Overview
This project predicts house prices using machine learning. Multiple regression models are compared, and the best one is selected based on **RMSE**. The final model is a **Lasso regression pipeline** with preprocessing, including:

- Imputation for missing values
- Scaling numeric features
- One-hot encoding categorical features

A **Streamlit interface** allows users to easily upload CSV files for predictions and download a ready-to-submit `submission.csv` for Kaggle.

## Features

### Regression Models Compared
- Random Forest Regressor
- HistGradientBoosting Regressor
- Ridge Regression
- Lasso Regression

### Preprocessing Pipeline
- Handles missing values:  
  - Numeric: mean  
  - Categorical: most frequent
- Feature scaling
- One-hot encoding
- Automatic selection of the best model based on RMSE
- Save and load the trained pipeline (`model.pkl`)

### Streamlit Interface
- Upload test CSV files
- Generate predictions
- Download submission-ready CSV

## Installation

### 1️⃣ Clone the repository
```bash
git clone <your-repo-url>
cd HOUSE_PRICES
```

### 2️⃣ Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing:
```bash
pip install pandas numpy scikit-learn joblib streamlit
```

## Usage

### 1️⃣ Train the Model
```bash
python src/train.py
```
- Trains multiple models
- Selects the best one (Lasso by default)
- Saves the pipeline as `model.pkl`

### 2️⃣ Predict with Test Data
```bash
python src/evaluate.py --input data/test.csv --output submission.csv
```
- Produces predictions using the trained pipeline

### 3️⃣ Streamlit Interface
```bash
python -m streamlit run app.py
```
- Upload a CSV file with house features
- View predictions in the browser
- Download `submission.csv` ready for Kaggle

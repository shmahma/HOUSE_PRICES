House Prices Prediction
Overview

This project predicts house prices using machine learning. It compares multiple regression models and selects the best one based on RMSE. The final model is a Lasso regression pipeline with preprocessing steps, including:

Imputation for missing values

Scaling numeric features

One-hot encoding categorical features

A Streamlit interface allows easy uploading of CSV files for predictions and downloading ready-to-submit submission.csv files for Kaggle.

Features

Compare multiple regression models:

Random Forest Regressor

HistGradientBoosting Regressor

Ridge Regression

Lasso Regression

Preprocessing pipeline handles:

Missing values (mean for numeric, most frequent for categorical)

Feature scaling

One-hot encoding

Automatic selection of the best model (lowest RMSE)

Save and load the trained pipeline (model.pkl)

Streamlit interface for:

Uploading test CSV files

Generating predictions

Downloading submission-ready CSV

Installation

Clone the repository:

git clone <your-repo-url>
cd HOUSE_PRICES


Create a virtual environment (recommended):

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux


Install dependencies:

pip install -r requirements.txt


If you don’t have requirements.txt, install manually:

pip install pandas numpy scikit-learn joblib streamlit

Usage
1️⃣ Train the Model
python src/train.py


This trains multiple models, selects the best one (Lasso by default), and saves the pipeline as model.pkl.

2️⃣ Predict with Test Data
python src/evaluate.py --input data/test.csv --output submission.csv


Produces predictions using the trained pipeline.

3️⃣ Streamlit Interface

Run the Streamlit app:

python -m streamlit run app.py


Upload a CSV file with house features

View predictions in the browser

Download submission.csv ready for Kaggle
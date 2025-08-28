# Heart Disease Prediction Project

This project is a modular machine learning pipeline to predict heart disease using Python, Kedro, and DVC. It demonstrates best practices in reproducible ML, including data versioning, modular code, and automated pipelines.

## Features
- Exploratory Data Analysis (EDA)
- Feature preprocessing and scaling
- Multiple ML models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Advanced pipeline using Kedro nodes and datasets
- Model versioning and dataset versioning with DVC
- Reproducible training and evaluation
- Easy extension for future ML tasks (e.g., CV/NLP)

## Project Structure

project/
├── data/ # DVC-tracked datasets and models
├── src/ # Modular source code
│ └── heart_disease_project/
│ ├── nodes/ # Kedro nodes
│ ├── pipelines/
│ └── hooks.py
├── conf/ # Kedro configurations
├── tests/ # Unit tests with pytest
├── README.md
├── requirements.txt
└── .gitignore


Install Python dependencies:
pip install -r requirements.txt

Install DVC dependencies:
pip install dvc kedro kedro-datasets

Pull data and model versions (DVC):
dvc pull

Run the Kedro pipeline:(have to run from folder heart_disease_project)
kedro run

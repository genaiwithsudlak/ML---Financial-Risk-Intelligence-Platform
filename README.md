# ML — Financial Risk Intelligence Platform

## Project Overview

The Financial Risk Intelligence Platform is an end-to-end machine learning project developed to detect credit card fraud using a production-ready, modular, and scalable architecture. The system is designed with MLOps principles, ensuring reproducibility, maintainability, and extensibility across the entire lifecycle.

The project is trained on multiple large-scale financial transaction datasets to ensure robustness, generalization, and real-world applicability.

---

## Key Features

### 1. Data Preprocessing Pipeline

* Comprehensive data cleaning and validation
* Handling missing values and inconsistent entries
* Categorical encoding using industry-standard encoding techniques
* Feature scaling and normalization
* Time-based feature extraction from transactional timestamps
* Customer and merchant demographic feature engineering
* Fully modular preprocessing using `sklearn.Pipeline`

### 2. Automated Exploratory Data Analysis (EDA)

* Target variable distribution analysis
* Fraud vs non-fraud comparison plots
* Time-based fraud trend visualizations
* Distribution analysis for key numerical features
* Relationship analysis between demographics, merchant attributes, and fraud behavior
* Automated plot generation and saving under:
  `reports/fraud/creditcard/`
* Optional full YData Profiling report for deeper insights

### 3. Feature Engineering

* Customer age calculation
* Time-derived features: hour, day, month, year, day-of-week
* Weekend and seasonal patterns
* Merchant vs customer location features
* Risk-based feature development (merchant type, category patterns)

### 4. Machine Learning Pipeline

* Train/test split with balanced sampling strategies
* End-to-end pipeline combining preprocessing and modeling
* Model development using:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Hyperparameter tuning via GridSearch and optionally Optuna

### 5. Model Evaluation

* ROC-AUC
* Precision, Recall, and F1-score
* Confusion matrix analysis
* Precision-Recall curve
* Threshold optimization and calibration

### 6. MLflow Integration

* Tracks model parameters, metrics, artifacts, and preprocessing components
* Supports experiment comparison
* Enables reproducible model development through logged pipelines

### 7. Hydra Configuration Management

* Centralized configuration via YAML files
* Modular configuration for model, data, and feature settings
* Seamless experiment management using Hydra overrides

### 8. Jupyter Notebook Support

* EDA notebook for detailed analysis
* Model evaluation notebook for comprehensive performance review

---

## Installation

### 1. Clone the Repository

```
git clone <repository_url>
cd ML---Financial-Risk-Intelligence-Platform
```

### 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## How to Run

### Run Preprocessing

```
python src/data/preprocessing.py
```

### Run EDA

```
python src/eda/eda_script.py
```

### Train Model

```
python src/models/train.py
```

### Optional: Launch MLflow UI

```
mlflow ui
```

---

## Model Performance

The platform includes multiple machine learning models to ensure optimal fraud detection. Model performance is evaluated using standard classification metrics and logged through MLflow for comparison. Detailed evaluations are available in the `evaluation.ipynb` notebook.

---

## Author

**Sudarshan Lakshate**
A dedicated technologist with expertise in machine learning, data analytics, SQL, and end-to-end MLOps-driven project development. Passionate about building scalable and int

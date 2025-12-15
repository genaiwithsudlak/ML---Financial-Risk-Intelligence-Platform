# 📘 Financial Risk Intelligence Platform - Recreation Guide

**Welcome!** This guide is designed to help you understand exactly what we built and **how to build it again from scratch**. Think of this as the "Master Recipe" for your project.

---

## 🏗️ 1. What Did We Build?

We built a **Machine Learning Platform** that acts like a highly intelligent bank security guard. It looks at different types of financial activities (credit cards, loans, transfers) and instantly decides: *"Is this risky?"*

### The 6 Core "Brains" (Models)
1.  **Credit Card Fraud**: Catches suspicious swipes (e.g., $500 at 3 AM).
2.  **IEEE-CIS Fraud**: Catches complex online identity theft.
3.  **PaySim Fraud**: Catches mobile money laundering.
4.  **Give Me Some Credit**: Predicts if someone will go broke in 2 years.
5.  **Home Credit**: Predicts if someone will default on a house loan.
6.  **Lending Club**: Predicts if a peer-to-peer loan will be paid back.

---

## 🛠️ 2. How to Recreate This Project (Step-by-Step)

If you deleted everything today, here is how you would rebuild it, step by step.

### Step 1: The Setup (The Foundation)
You need a "kitchen" to cook in.
1.  **Create a Folder**: `mkdir Financial-Risk-Platform`
2.  **Create a Virtual Environment** (keeps your tools organized):
    ```bash
    python -m venv venv
    source venv/bin/activate  # (or venv\Scripts\activate on Windows)
    ```
3.  **Install Tools**: Create a `requirements.txt` file listing libraries like `pandas`, `xgboost`, `fastapi`, `streamlit`. Then run:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Organize the Files (The "Blueprint")
We used a professional structure known as "Cookiecutter Data Science".
*   `data/`: Where raw CSV files go.
*   `src/`: Where all your python code lives.
    *   `src/data`: Scripts to load CSVs.
    *   `src/preprocessing`: Scripts to clean data (fill missing values, turn text into numbers).
    *   `src/models`: Scripts to train the AI.
    *   `src/pipelines`: Scripts that connect Data -> Preprocessing -> Model.
    *   `src/api`: The web server (FastAPI).
    *   `src/ui`: The dashboard (Streamlit).

### Step 3: Building a Pipeline (The "Cooking Process")
For *each* of the 6 datasets, we did the same 3 things:
1.  **Load**: Write a function in `src/data/make_dataset.py` to read the CSV.
2.  **Clean**: Write a class in `src/preprocessing/` (e.g., `fraud_preprocessing.py`) to handle messy data.
    *   *Example*: We calculated "Age" from "Date of Birth".
3.  **Train**: Write a script in `src/models/` to teach XGBoost or LightGBM.
    *   *Example*: `train_model(data)` -> saves a `.pkl` file (the trained brain).

### Step 4: Serving the Models (The "Waiter")
We need a way for users to talk to the models.
*   We created `src/api/main.py` using **FastAPI**.
*   It has endpoints like `/predict/credit_card`. You send it JSON data, and it sends back a "Risk Score".

### Step 5: The Interface (The "Menu")
We built a website using **Streamlit** in `src/ui/app.py`.
*   It has 6 tabs.
*   When you click "Predict", it talks to the API (The Waiter) to get the answer.

### Step 6: Testing (The "Quality Check")
We wrote tests in `tests/` to make sure:
1.  Data loads correctly.
2.  The API doesn't crash.
3.  The pipelines run from start to finish.
*   Run them with `pytest`.

---

## 🤖 3. The "Magic" Explained

### Why XGBoost/LightGBM?
These are "Gradient Boosting" models. Imagine asking 100 smart students to solve a problem. The first student makes a guess. The second student focuses *only on what the first got wrong*. The third focuses on the remaining errors. By the end, they are super accurate.

### Why Docker?
It wraps your entire project (code + tools) in a "box". You can hand this box to anyone (or any server), and it will run exactly the same way.

### Why MLflow?
It's a diary for your experiments. Every time you train a model, it records the settings (hyperparameters) and the score (Accuracy).

---

## 🚀 4. Cheat Sheet Commands

**Run the API:**
```bash
uvicorn src.api.main:app --reload
```

**Run the Dashboard:**
```bash
streamlit run src/ui/app.py
```

**Run Tests:**
```bash
pytest tests/
```

**Retrain a Model:**
```bash
python src/pipelines/lendingclub_pipeline.py
```

---

*Keep this guide safe! With these instructions, you can rebuild this complex system anytime.*

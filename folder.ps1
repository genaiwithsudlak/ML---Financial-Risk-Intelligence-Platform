# ============================================
#  Unified Risk & Fraud ML Project Structure
#  Auto-Generate Directories + __init__.py Files
# ============================================

$root = "unified_risk_fraud_ml"

# Helper: Create folder and optional __init__.py
function New-PyFolder($path, $addInit=$true) {
    if (!(Test-Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
    if ($addInit -and !(Test-Path "$path\__init__.py")) {
        New-Item -ItemType File -Path "$path\__init__.py" | Out-Null
    }
}

# -------------------------------
# Create root project directory
# -------------------------------
New-PyFolder $root $false

# -------------------------------
# data directories
# -------------------------------
$dataDirs = @(
    "data/raw/fraud",
    "data/raw/loan",
    "data/interim/fraud",
    "data/interim/loan",
    "data/processed/fraud",
    "data/processed/loan",
    "data/external"
)

foreach ($dir in $dataDirs) {
    New-PyFolder "$root/$dir" $false
}

# -------------------------------
# notebooks
# -------------------------------
$notebookDirs = @(
    "notebooks/fraud",
    "notebooks/loan"
)

foreach ($dir in $notebookDirs) {
    New-PyFolder "$root/$dir" $false
}

# Create placeholder notebooks
$placeholderNotebooks = @{
    "notebooks/fraud/EDA_fraud_cc.ipynb" = "";
    "notebooks/fraud/EDA_fraud_ieee.ipynb" = "";
    "notebooks/fraud/EDA_fraud_synthetic.ipynb" = "";
    "notebooks/fraud/Fraud_Features_Analysis.ipynb" = "";

    "notebooks/loan/EDA_gmsc.ipynb" = "";
    "notebooks/loan/EDA_home_credit.ipynb" = "";
    "notebooks/loan/EDA_lendingclub.ipynb" = "";
    "notebooks/loan/Loan_Features_Analysis.ipynb" = "";

    "notebooks/Model_Comparison.ipynb" = "";
}

foreach ($nb in $placeholderNotebooks.Keys) {
    New-Item -ItemType File -Path "$root/$nb" | Out-Null
}

# -------------------------------
# src package structure
# -------------------------------
$srcPackages = @(
    "src",
    "src/config",
    "src/utils",
    "src/preprocessing",
    "src/models",
    "src/models/fraud",
    "src/models/loan",
    "src/pipelines",
    "src/evaluation",
    "src/api"
)

foreach ($folder in $srcPackages) {
    New-PyFolder "$root/$folder"
}

# -------------------------------
# src/config placeholder yaml
# -------------------------------
New-Item -ItemType File -Path "$root/src/config/fraud_config.yaml" | Out-Null
New-Item -ItemType File -Path "$root/src/config/loan_config.yaml" | Out-Null

# -------------------------------
# Placeholder python files
# -------------------------------
$pyFiles = @(
    # utils
    "src/utils/logger.py",
    "src/utils/io_utils.py",
    "src/utils/metrics.py",
    "src/utils/common_preprocessing.py",

    # preprocessing
    "src/preprocessing/fraud_preprocessing.py",
    "src/preprocessing/loan_preprocessing.py",

    # models
    "src/models/fraud/train_fraud_model.py",
    "src/models/fraud/fraud_lgbm.py",
    "src/models/loan/train_loan_model.py",
    "src/models/loan/loan_lgbm.py",

    # pipelines
    "src/pipelines/fraud_pipeline.py",
    "src/pipelines/loan_pipeline.py",
    "src/pipelines/pipeline_runner.py",

    # evaluation
    "src/evaluation/fraud_eval.py",
    "src/evaluation/loan_eval.py",

    # api
    "src/api/fastapi_server.py",
    "src/api/fraud_predict.py",
    "src/api/loan_predict.py"
)

foreach ($file in $pyFiles) {
    New-Item -ItemType File -Path "$root/$file" | Out-Null
}

# -------------------------------
# models, logs, reports structure
# -------------------------------
$otherDirs = @(
    "models/fraud",
    "models/loan",
    "models/scaler_encoder_objects",
    "logs/fraud",
    "logs/loan",
    "reports/fraud",
    "reports/loan",
    "configs",
    "tests"
)

foreach ($d in $otherDirs) {
    New-PyFolder "$root/$d" $false
}

# -------------------------------
# Create placeholder config yaml
# -------------------------------
New-Item -ItemType File -Path "$root/configs/fraud.yaml" | Out-Null
New-Item -ItemType File -Path "$root/configs/loan.yaml" | Out-Null

# -------------------------------
# Create main files
# -------------------------------
New-Item -ItemType File -Path "$root/main.py" | Out-Null
New-Item -ItemType File -Path "$root/README.md" | Out-Null
New-Item -ItemType File -Path "$root/requirements.txt" | Out-Null
New-Item -ItemType File -Path "$root/.gitignore" | Out-Null

Write-Host "✅ Project structure created successfully!"

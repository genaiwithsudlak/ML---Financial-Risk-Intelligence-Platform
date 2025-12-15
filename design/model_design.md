# ML Model Design Document 

> **Purpose**: This document captures all decisions, assumptions, and strategies involved in designing a production-ready Machine Learning model. It acts as a single source of truth for engineers, reviewers, and stakeholders.

---

## 1. Problem Definition

### 1.1 Business Objective

* **Goal**: Detect and prevent fraudulent credit card transactions in real-time to minimize financial loss for the institution and customers.
* **Impact**: Reduce chargeback costs and improve trust, while maintaining a low friction experience (low false declines) for genuine users.

### 1.2 ML Problem Formulation

* **Problem Type**: Binary Classification (Fraud=1 vs Legitimate=0).
* **Prediction Target**: `isFraud` (Probability score 0.0 - 1.0).
* **Prediction Horizon**: Immediate (Real-time assessment at the moment of transaction).
* **Real-time or Batch**: Real-time Inference (API based).

### 1.3 Success Criteria

* **Business KPI**: 
    * Fraud Loss Savings ($)
    * Fraud Detection Rate (Recall) > 90%
    * False Positive Rate (FPR) < 0.5% (to minimize customer insult)
* **ML Metric Thresholds**:
    * ROC-AUC > 0.95
    * Precision-Recall AUC (PR-AUC) > 0.80
    * Precision @ Recall 90% > 0.30

---

## 2. Assumptions & Constraints

### 2.1 Business Constraints

* **Cost of false positives**: High (Customer churn, support calls, reputational damage).
* **Cost of false negatives**: High (Direct financial loss, chargeback fees).
* **Regulatory / compliance requirements**: PCI-DSS compliance for data handling. GDPR for user data privacy. Explainability required for declined transactions (why was this blocked?).

### 2.2 System Constraints

* **Latency SLA**: < 200ms p99 latency per prediction request.
* **Throughput**: Support peak of 1000 transactions per second (TPS).
* **Compute (CPU/GPU)**: CPU-optimized inference for cost-efficiency.
* **Memory limits**: Model artifact < 500MB to fit in standard container memory.

---

## 3. Data Design

### 3.1 Data Sources

| Source | Type | Frequency |
| ------ | ---- | --------- |
| Transaction Logs | Structured (SQL) | Real-time |
| Customer Profile | User Attributes | Daily Batch Update |
| Merchant Database | Risk Scores/Categories | Weekly Batch |
| Device Fingerprint | Metadata | Real-time |

### 3.2 Feature Design

#### 3.2.1 Feature Categories

* **Static features**: Customer Age, Account Age, Merchant Category Code (MCC).
* **Dynamic / behavioral features**: Transaction Amount, Time of Day, Day of Week.
* **Aggregated features**: 
    * Rolling count of transactions in last 1h, 24h.
    * Average transaction amount in last 7 days vs current amount.
    * Number of failed attempts in last 30 mins.

#### 3.2.2 Feature Availability (Inference Contract)

| Feature | Source | Available at Inference (Yes/No) |
| ------- | ------ | ------------------------------- |
| Transaction Amount | Request Payload | Yes |
| Merchant ID | Request Payload | Yes |
| Last 1h Tx Count | Feature Store (Redis) | Yes |
| Account Age | User DB | Yes |
| Future Chargeback Status | Label DB | No (Target Leakage) |

> **Rule**: Features unavailable at inference time must not be used.

### 3.3 Feature Freshness

* **Real-time features**: Current transaction details (Amount, Geo-location).
* **Batch features**: Customer risk profile (updated daily).
* **Near-real-time aggregates**: Velocity counters (updated via stream processing).

### 3.4 Missing Value Strategy

* **Imputation method**: 
    * Numerical: Median imputation (robust to outliers).
    * Categorical: New category 'Unknown'.
* **Default handling**: XGBoost/LightGBM handle missing values natively (assigned to default direction).

### 3.5 Feature Store Architecture (Advanced)

*   **Offline Store**:
    *   **Technology**: Parquet/Iceberg on S3/GCS.
    *   **Purpose**: Generating point-in-time correct training datasets (preventing partial leakage).
*   **Online Store**:
    *   **Technology**: Redis / DynamoDB.
    *   **Purpose**: Serving low-latency features at <10ms p99 to the inference service.
*   **Feature Sync**:
    *   Stream processing (Flink/Spark Streaming) updates Online Store in near-real-time.
    *   Batch jobs (Airflow) populate Offline Store nightly.

---

## 4. Leakage Prevention Strategy

### 4.1 Temporal Leakage

* **Split Strategy**: Strictly time-based splitting. 
    * Train: Jan - Sep
    * Validation: Oct
    * Test: Nov - Dec
* **Rule**: Validation/Test sets must strictly follow Training data in time.

### 4.2 Feature Leakage

* Explicit exclusion of 'Fraud Label', 'Chargeback Date', or 'Transaction Status' (Approved/Declined) from input features.

### 4.3 Aggregation Safety

* Aggregates (e.g., "Avg daily spend") must be calculated using a window that strictly precedes the current transaction time. No "current day total" including the current future transaction.

---

## 5. Model Architecture Design

### 5.1 Model Selection Rationale

| Constraint     | Choice |
| -------------- | ------ |
| **Explainability** | High importance (Regulatory). Tree-based models offer feature importance/shap values. |
| **Latency**        | Critical. Trees are faster than large NNs on CPU. |
| **Data Size**      | Medium-Large (Millions of rows). Trees scale well. |

### 5.2 Candidate Models Considered

* **Baseline model**: Logistic Regression (Simple, interpretable, fast).
* **Advanced models**: XGBoost, LightGBM, CatBoost (SOTA for tabular data), Random Forest.

### 5.3 Final Model Choice

* **Model type**: **XGBoost** (or LightGBM).
* **Justification**: Best trade-off between accuracy and inference speed. Handles non-linear relationships and interactions (e.g., High Amount + Night Time) better than linear models. Native missing value handling.

---

## 6. Training Strategy

### 6.1 Data Splitting Strategy

* **Split type**: Time-based (OOT - Out of Time validation).
* **Train range**: Historical 12 months.
* **Validation range**: Next 1 month.
* **Test range**: Subsequent 1 month.

### 6.2 Class Imbalance Handling

* **Strategy**: 
    * **scale_pos_weight**: Tuning the positive class weight in XGBoost/LGBM.
    * **SMOTE**: Synthetic Minority Over-sampling (typically used in training only, if weighting isn't enough).
    * **Downsampling**: Randomly downsampling majority class (Legitimate) to 10:1 ratio.

### 6.3 Hyperparameter Strategy

* **Search method**: Bayesian Optimization (Optuna).
* **Validation approach**: Time-Series Cross-Validation (expanding window).

### 6.4 Training Configuration

* **Optimizer**: Adam (if NN) / Gradient Descent (Trees).
* **Learning rate**: 0.01 - 0.1 (tuned).
* **Batch size**: N/A for Trees (or chunk size).
* **Early stopping**: Stop if validation AUC doesn't improve for 50 rounds.

---

## 7. Evaluation Strategy

### 7.1 Offline Metrics

| Metric | Purpose |
| ------ | ------- |
| **ROC-AUC** | General discriminative power (rank ordering). |
| **Precision-Recall AUC** | Performance on minority class (Fraud). Critical for imbalanced data. |
| **Precision @ k** | Precision at top k% riskiest transactions. |

### 7.2 Threshold Selection

* **Strategy**: Maximize Recall while keeping Precision > P (e.g., 20%).
* **Decision thresholds**: 
    * Score > 0.9: Auto-Decline
    * 0.7 < Score < 0.9: Step-up Auth (SMS/2FA) (Review)
    * Score < 0.7: Approve

### 7.3 Slice-Based Evaluation

* **Segments**:
    * New vs Existing Users
    * High vs Low Value Transactions
    * International vs Domestic
    * Card Type (Debit/Credit/Premium)

### 7.4 Stress & Robustness Tests

* **Null injection**: Test inference when top 3 features are missing.
* **Out-of-distribution**: Evaluate on holiday season data (Black Friday) when training on standard months.

### 7.5 Advanced Label Strategy: The "Censored Label" Problem

*   **Problem**: We only know the label (Fraud/Legit) for *approved* transactions. Declined transactions have no labels (we don't know if we correctly stopped fraud or stopped a genuine user).
*   **Strategy**:
    *   **Control Group (Random Holdback)**: Randomly approve 1-5% of transactions that the model *would have declined*.
    *   **Purpose**: Gather ground-truth labels for the "risky" population to prevent model drift into blindness.
    *   **Warning**: This incurs short-term fraud loss for long-term model health.

---

## 8. Deployment Design

### 8.1 Inference Mode

* **Real-time API**: HTTP endpoint (FastAPI) receiving JSON payload.

### 8.2 Serving Architecture

* **Model format**: ONNX (for fastest inference) or MLflow Packet (Python function).
* **Serving framework**: FastAPI containerized in Docker.

### 8.3 Fallback Strategy

* **Rule-based fallback**: If model times out (>200ms) or fails, apply static rules (e.g., Decline if Amount > $10k, else Approve).
* **Previous model fallback**: Keep previous stable model version active (Shadow mode or A/B flip).

---

## 9. Monitoring & Observability

### 9.1 Data Drift Monitoring

| Feature | Drift Test | Threshold |
| ------- | ---------- | --------- |
| Transaction Amount | PSI (Population Stability Index) | > 0.1 |
| Merchant Category | Chi-Square Test | p-value < 0.05 |

### 9.2 Performance Monitoring

* **Metrics tracked**: Real-time Fraud Rate, Decline Rate. Lagged Label Metrics (Precision/Recall after chargebacks arrive).
* **Alert thresholds**: If Decline Rate spikes > 50% deviation from norm -> Trigger P1 Alert.

### 9.3 Latency & System Metrics

* **p95 latency**: Tracked via Prometheus/Grafana.
* **Error rate**: Non-200 responses.

---

## 10. Retraining Strategy

### 10.1 Retraining Triggers

* **Scheduled**: Monthly.
* **Drift-based**: Significant drift in key features (PSI > 0.2).
* **Performance-based**: Discovery of new fraud attack vector (Recall drops).

### 10.2 Retraining Frequency

* **Cadence**: Monthly full retrain. Weekly incremental updates (if online learning supported, otherwise monthly).

### 10.3 Model Versioning

* **Scheme**: Semantic (v1.0.0).
* **Registry**: MLflow Model Registry (Stage: Staging -> Production).
* **Rollback**: One-click rollback to previous version in registry.

---

## 11. Failure Handling

### 11.1 Data Failures

* **Schema validation**: Pydantic models in FastAPI to reject malformed requests.
* **Null spike handling**: Default imputation logic active.

### 11.2 Prediction Failures

* **Low confidence**: Flag for manual review if score is borderline (0.45 - 0.55) - optional.

### 11.3 System Failures

* **Service downtime**: Load balancer redirects to hot-standby region or fallback rule engine.

### 11.4 Explainability & Compliance (Deep Dive)

*   **Global Explainability**:
    *   **SHAP/LIME**: Batch jobs run daily on a sample of predictions to understand top driving features globally.
*   **Local Explainability (Reason Codes)**:
    *   **Requirement**: Every declined transaction must have a reason.
    *   **Implementation**: Fast TreeSHAP (approx) at inference time to return top 3 contributing features (e.g., "High Transaction Amount", "Location Mismatch").
*   **Auditing**:
    *   Log all model inputs, outputs, and software versions for 7 years (regulatory requirement).

---

## 12. Risks & Mitigations

| Risk | Mitigation |
| ---- | ---------- |
| **Bias/Fairness** | Model discriminating against certain demographics (age/location). **Mitigation**: Fairness auditing (Demographic Parity tests). |
| **Adversarial Attacks** | Fraudsters probing decision boundaries. **Mitigation**: Rate limiting, obfuscating exact reason for decline, frequent retraining. |
| **Cold Start** | New customers have no history. **Mitigation**: Use global aggregates and explicit "New User" features/rules. |

---

## 13. Model Design Sign-Off

* **Author**: Antigravity (AI Assistant)
* **Reviewers**: User (Lead Engineer)
* **Date**: 2025-12-15
* **Approved Version**: v1.0-Draft

---

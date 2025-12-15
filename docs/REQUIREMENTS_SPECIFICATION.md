# Business Requirements Document (BRD)

**Project Name:** Financial Risk Intelligence Platform  
**Version:** 1.0  
**Status:** Final  
**Author:** Sudarshan Lakshate  
**Reviewer:** Sudarshan Lakshate  
**Date:** 2025-12-15  

---

## 1. Executive Summary
The **Financial Risk Intelligence Platform** is an enterprise-grade Machine Learning solution designed to detect, predict, and mitigate financial risks across multiple domains. By leveraging advanced gradient boosting algorithms (XGBoost/LightGBM), the platform provides real-time risk assessments for credit card transactions, loan applications, and mobile money transfers.

## 2. Project Scope

### 2.1 In-Scope
*   **Fraud Detection**: Real-time identification of fraudulent activities in:
    *   Credit Card Transactions (Anomalies, Theft).
    *   IEEE-CIS Dataset (Identity & Transaction Fraud).
    *   PaySim Dataset (Mobile Money Laundering).
*   **Credit Default Prediction**: Assessment of borrower capability for:
    *   General Consumer Loans (Give Me Some Credit).
    *   Unbanked Population (Home Credit).
    *   Peer-to-Peer Lending (Lending Club).
*   **Deployment**: Dockerized inference API and interactive Dashboard.

### 2.2 Out-of-Scope
*   Integration with live banking mainframes (Mock API provided).
*   Real-time retraining (Batch retraining supported).

## 3. Stakeholders
| Role | Name | Responsibilities |
| :--- | :--- | :--- |
| **Project Lead / Author** | **Sudarshan Lakshate** | Architecture, Implementation, Documentation. |
| **Reviewer** | **Sudarshan Lakshate** | Code Review, QA, Final Approval. |
| **End Users** | Risk Analysts | Monitor dashboard and review flagged cases. |

## 4. Functional Requirements

### 4.1 Usage Scenarios (User Stories)
*   **AS A** Risk Analyst, **I WANT** to input transaction details into a dashboard, **SO THAT** I can get an immediate probability of fraud.
*   **AS A** Loan Officer, **I WANT** to see the top contributing factors to a rejection, **SO THAT** I can explain decisions to applicants (Explainability).

### 4.2 Data Requirements
*   System must ingest CSV/Parquet data formats.
*   Must handle missing values (Imputation) and categorical variables (Encoding) automatically.

### 4.3 Performance Requirements
*   **Latnecy**: API response time < 200ms for single prediction.
*   **Availability**: 99.9% simulation uptime via Docker containers.

## 5. Risk & Compliance
*   **Data Privacy**: No real PII (Personally Identifiable Information) used; all datasets are anonymized public benchmarks.
*   **Auditability**: All model runs, parameters, and metrics must be logged via MLflow.

---
*Confidential - Internal Use Only*

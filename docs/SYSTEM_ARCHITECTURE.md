# System Architecture Document (HLD)

**Project Name:** Financial Risk Intelligence Platform  
**Version:** 1.0  
**Author:** Sudarshan Lakshate  
**Reviewer:** Sudarshan Lakshate  

---

## 1. Introduction
This document outlines the high-level architecture of the Financial Risk Intelligence Platform. It details the system components, their interactions, and the data flow from ingestion to inference.

## 2. System Overview
The platform follows a **Microservices-based architecture** (conceptually) containerized via Docker. It separates the Model Training pipeline from the Inference API and User Interface.

### 2.1 High-Level Diagram
```mermaid
graph TD
    User[Risk Analyst] -->|Interaction| UI[Streamlit Dashboard]
    UI -->|HTTP POST| API[FastAPI Inference Service]
    
    subgraph "Inference Layer"
        API -->|Load| Registry[Model Registry (MLflow)]
        API -->|Preprocess| FeatureStore[Feature Engineering Logic]
    end
    
    subgraph "Training Layer"
        RawData[(Raw Data)] -->|ETL| Pipeline[Training Pipeline]
        Pipeline -->|Train| XGB[XGBoost/LGBM Model]
        XGB -->|Log| MLflow[MLflow Tracking Server]
        XGB -->|Save Artifact| Registry
    end
    
    MLflow -->|Metrics| Monitor[Performance Monitoring]
```

## 3. Component Description

### 3.1 Data Ingestion & Processing
*   **Source**: CSV files (Batch ingestion).
*   **Logic**: `src/data` and `src/preprocessing`.
*   **Responsibility**: Validation, Cleaning, Transformation (Ordinal Encoding, Scaling).

### 3.2 Model Training Engine
*   **Technology**: Python, Scikit-Learn, XGBoost, Optuna.
*   **Function**: Automated hyperparameter tuning and model serialization.
*   **Output**: Compressed `.pkl` or `.json` model artifacts.

### 3.3 Experiments & Registry (MLflow)
*   **Server**: Local MLflow server (extensible to remote).
*   **Artifacts**: Stores serialized models and training metadata.

### 3.4 Inference API (FastAPI)
*   **Endpoints**: RESTful endpoints separated by domain (e.g., `/predict/credit_card`).
*   **Scalability**: Stateless design, horizontally scalable via Docker Swarm/K8s (future).
*   **Monitoring**: Prometheus middleware for latency and request tracking.

### 3.5 User Interface (Streamlit)
*   **Type**: Single Page Application (SPA).
*   **Features**: Dynamic forms, Real-time probability visualization, interpretable results.

## 4. Technology Stack
| Layer | Technology |
| :--- | :--- |
| **Language** | Python 3.9+ |
| **ML Frameworks** | XGBoost, LightGBM, Scikit-Learn |
| **Orchestration** | MLflow |
| **API** | FastAPI |
| **Frontend** | Streamlit |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |

---
*Confidential - Internal Use Only*

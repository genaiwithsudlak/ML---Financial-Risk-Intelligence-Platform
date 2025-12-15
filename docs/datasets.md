# Dataset Documentation

This document explicitly details the datasets identified for the Financial Risk Intelligence Platform, covering both Fraud Detection and Loan Default prediction.

---

## 🛡️ Fraud Detection Datasets

### 1. Credit Card Transactions (Priyam Choksi)
*   **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)
*   **Description**: A detailed record of credit card transactions, useful for identifying fraudulent patterns based on transaction metadata and demographics.
*   **Key Columns**:
    *   `trans_date_trans_time`: Timestamp.
    *   `cc_num`: Masked credit card number.
    *   `merchant`: Merchant name.
    *   `category`: Transaction category (e.g., grocery, entertainment).
    *   `amt`: Transaction amount in USD.
    *   `is_fraud`: **Target Variable** (0=Legit, 1=Fraud).
    *   `lat`, `long`: Cardholder location.
    *   `merch_lat`, `merch_long`: Merchant location.

### 2. IEEE-CIS Fraud Detection
*   **Link**: [Kaggle Competition](https://www.kaggle.com/c/ieee-fraud-detection)
*   **Description**: A complex, real-world e-commerce dataset from Vesta Corporation. It features a wide array of masked features and requires joining transaction and identity tables.
*   **Key Columns**:
    *   `TransactionID`: Unique ID (Join Key).
    *   `isFraud`: **Target Variable**.
    *   `TransactionDT`: Time delta.
    *   `TransactionAmt`: Amount.
    *   `ProductCD`: Product code.
    *   `card1` - `card6`: Payment card info.
    *   `addr1`, `addr2`: Address info.
    *   `P_emaildomain`: Purchaser email.
    *   `V1` - `V339`: Vesta-engineered rich features (masked).

### 3. Synthetic Financial Datasets (PaySim)
*   **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
*   **Description**: A synthetic dataset simulating mobile money transactions, designed to fill the gap of public financial datasets.
*   **Key Columns**:
    *   `step`: Maps to 1 hour of time.
    *   `type`: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER.
    *   `amount`: Transaction amount.
    *   `nameOrig`, `nameDest`: Origin and Destination IDs.
    *   `isFraud`: **Target Variable**.
    *   `isFlaggedFraud`: Business rule flag (>200k transfer).

---

## 💸 Loan Default Datasets

### 1. Give Me Some Credit
*   **Link**: [Kaggle Competition](https://www.kaggle.com/c/GiveMeSomeCredit)
*   **Description**: Improve on the state of the art in credit scoring by predicting the probability that somebody will experience financial distress in the next two years.
*   **Key Columns**:
    *   `SeriousDlqin2yrs`: **Target Variable** (90 days past due).
    *   `RevolvingUtilizationOfUnsecuredLines`: Credit utilization %.
    *   `Age`: Borrower age.
    *   `DebtRatio`: Monthly debt payments / monthly income.
    *   `MonthlyIncome`: Monthly income.

### 2. Home Credit Default Risk
*   **Link**: [Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk)
*   **Description**: Predict clients' repayment abilities using telecom and transactional information. Contains multiple relational tables.
*   **Key Columns** (Main Application Table):
    *   `SK_ID_CURR`: Unique Loan ID.
    *   `TARGET`: **Target Variable** (1=Default).
    *   `NAME_CONTRACT_TYPE`: Cash loans or Revolving loans.
    *   `AMT_INCOME_TOTAL`: Income.
    *   `AMT_CREDIT`: Loan amount.

### 3. Lending Club Loan Data
*   **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
*   **Description**: Historical data on loans issued by LendingClub. Includes loan status and latest payment information.
*   **Key Columns**:
    *   `loan_status`: **Target Variable** (Current, Charged Off, Fully Paid).
    *   `int_rate`: Interest Rate.
    *   `installment`: Monthly payment.
    *   `annual_inc`: Annual income.
    *   `dti`: Debt-to-Income ratio.
    *   `fico_range_low`, `fico_range_high`: FICO scores.
    *   `purpose`: Loan purpose.

# 📊 Vendor Invoice Intelligence System
> **AI-Driven Freight Cost Prediction & Invoice Risk Flagging**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest%20%7C%20XGBoost-orange.svg)

---

## 📌 Project Overview
This project implements an **end-to-end machine learning ecosystem** designed to modernize finance and logistics operations. By leveraging historical data, the system automates two critical business workflows:

1.  **Freight Cost Estimation:** Predicts expected freight costs based on invoice value and item quantity to improve budgeting accuracy.
2.  **Risk Flagging:** Automatically detects high-risk invoices that deviate from normal patterns, reducing financial leakage and fraud.

---

## 📑 Table of Contents
* [Business Objectives](#-business-objectives)
* [Data Sources & EDA](#-data-sources--eda)
* [Model Architecture](#-model-architecture)
* [Project Structure](#-project-structure)
* [Installation & Usage](#-installation--usage)
* [Author & Contact](#-author--contact)

---

## 🎯 Business Objectives
* **Reduce Manual Workload:** Automate the initial screening of thousands of monthly invoices.
* **Prevent Overpayment:** Ensure freight charges align with historical norms.
* **Enhance Audit Accuracy:** Direct human auditors to high-probability "flagged" cases rather than random sampling.

---

## 📂 Data Sources & EDA
The system processes structured vendor data, including:
* **Features:** Invoice Quantity, Dollars, Freight Paid, Days from PO to Invoice, and Brand variety.
* **Analysis:** Extensive Exploratory Data Analysis (EDA) was performed to identify correlations between shipment size and shipping costs.

---

## 🤖 Model Architecture
| Task | Model Used | Primary Metric |
| :--- | :--- | :--- |
| **Freight Prediction** | Random Forest Regressor | MAE / R² Score |
| **Invoice Flagging** | XGBoost Classifier | Precision / Recall |

---

## 🏗 Project Structure
```bash
├── data/                       # Raw and processed datasets
├── freight_cost_prediction/    # Notebooks and scripts for regression
├── invoice_flagging/           # Notebooks and scripts for classification
│   ├── models/                 # Saved .pkl files (RandomForest, Scalers)
│   └── train.py                # Training logic
├── inference/                  # Prediction logic for production
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
├── app.py                      # Streamlit Web Application
└── README.md                   # Project documentation

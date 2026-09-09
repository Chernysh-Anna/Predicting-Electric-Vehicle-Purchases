
---

# 🚗 Will Buy Electric Vehicle (EV) — Tabular ML Pipeline

A machine learning pipeline for predicting EV adoption (`Will_Buy_EV`), featuring robust cross-validation, hyperparameter-configured gradient boosting models (**LightGBM**, **CatBoost**, and **XGBoost**), and automated overfitting diagnostics.

---

## 📊 Model Performance Summary

All models were evaluated using **5-Fold Stratified Cross-Validation** optimized for **ROC-AUC**.

| Model | Mean Train AUC | Mean Val AUC | Train-Val Gap | Fold Val Std |
| --- | --- | --- | --- | --- |
| **LightGBM** | 0.94734 | 0.94109 | 0.00626 | 0.00068 |
| **CatBoost** | 0.94320 | 0.94111 | 0.00209 | 0.00078 |
| **XGBoost** | 0.94560 | 0.94115 | 0.00445 | 0.00069 |

### 💡 Generalization Rule of Thumb

* **Healthy Fit:** A train-validation AUC gap under **`0.01 – 0.02`** (combined with early stopping and low fold-to-fold variance) indicates a robust model.
* **Overfitting Signature:** A massive gap accompanied by validation scores that swing wildly across different folds indicates memorization of noise rather than true pattern learning.
* *Our models exhibit an exceptionally stable validation standard deviation (`< 0.0008`) and tight gaps (`< 0.006`), confirming strong out-of-sample reliability.*

---

## 📂 Project Structure

```text
├── train.csv                               # Competition training data
├── test.csv                                # Competition test data
├── EV_Adoption_and_Range_Anxiety_Dataset.csv # Optional external supplementary data
├── pipeline.ipynb                          # Main exploratory data analysis & training notebook
└── README.md                               # Project documentation

```

---

## ⚙️ Installation & Quickstart

### 1. Requirements

Ensure you have Python 3.8+ installed along with the required gradient boosting and machine learning libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm catboost xgboost kaggle

```

### 2. Kaggle API Authentication (Optional for local execution)

To pull competition datasets directly via the terminal:

```bash
export KAGGLE_API_TOKEN="your_api_token_here"
kaggle competitions download -c playground-series-s6e9

```

### 3. Running the Pipeline

Execute the Jupyter notebook or python script to initiate feature engineering, model training loops, and evaluation metrics:

```bash
jupyter notebook pipeline.ipynb

```

---

## 🔍 Key Pipeline Features

1. **Robust Preprocessing:** Automated mapping of target labels (`Yes`/`No` to binary `1`/`0`) and strict handling of categorical types for optimal compatibility across libraries (especially string-casting enforcement for CatBoost).
2. **Multi-Model Benchmark:** Side-by-side evaluation of the three premier gradient boosting frameworks:
* **LightGBM:** Fast histogram-based splitting with regularization (`reg_alpha`, `reg_lambda`).
* **CatBoost:** Symmetric trees with native categorical feature handling.
* **XGBoost:** Highly regularized depth-wise growth.


3. **External Data Integration:** Optional ingestion of supplementary survey information to validate behavioral features like range anxiety and subsidy impact against synthetic competition boundaries.

---

## 📈 Results & Next Steps

* **Best Single Model:** **XGBoost** / **CatBoost** (tied closely at $\sim 0.94115$ Mean Val AUC).
* **Ensembling Potential:** Because individual model predictions are highly correlated yet slightly diverse, blending **LightGBM + CatBoost + XGBoost** via weighted averaging or stacking will likely yield a higher leaderboard score.
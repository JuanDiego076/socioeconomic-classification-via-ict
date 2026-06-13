# 🏠 Socioeconomic Classification via ICT Usage Patterns
### Predicción de Estrato Socioeconómico mediante Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Pipeline-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-009E73?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Dataset](https://img.shields.io/badge/Dataset-ENDUTIH_2024_(INEGI)-4A90D9?style=for-the-badge)](https://www.inegi.org.mx/programas/endutih/2024/)

---

## 🎯 Problem Statement

A household's access to ICT services — internet connectivity, TV subscriptions, number of smartphones, type of connection — is a strong proxy for its socioeconomic standing. This project asks: **can we predict a household's socioeconomic stratum purely from its technology profile?**

The answer has direct applications in public policy (subsidy targeting, digital inclusion programs) and commercial strategy (market segmentation, telecom product design).

**Dataset:** ENDUTIH 2024 (INEGI) — Mexico's national household survey on ICT availability and usage. **58,080 households**, multiclass target: `ESTRATO` (socioeconomic stratum).

---

## 📊 Results — 5-Model Comparison

All models evaluated with **10-fold Stratified Cross-Validation** on 80% training data, final metrics reported on a held-out 20% test set.

| Model | Accuracy | F1-score (macro) | Notes |
|---|---|---|---|
| **Bagging (DT base)** | **~97%** | **~97%** | 🏆 Best overall — robust to noise |
| **Random Forest** | **~97%** | **~97%** | 🥈 Comparable, with hyperparameter tuning |
| Decision Tree | ~95% | ~95% | Strong single-model baseline |
| XGBoost | ~93%+ | ~93%+ | Competitive; sensitive to feature scale |
| AdaBoost | ~70% | ~70% | Most affected by class imbalance & nulls |

**Key finding:** Ensemble methods based on decision trees dramatically outperform boosting approaches on this tabular dataset with mixed variable types and missing values — a pattern consistent with the broader XGBoost vs. Random Forest literature on heterogeneous tabular data.

---

## 🏛️ Architecture — Sklearn Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING PIPELINE                          │
│                                                                     │
│  Raw Features (ENDUTIH 2024)                                        │
│  58,080 households × ~100 ICT variables                             │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐   │
│  │   NUMERIC BRANCH    │    │      CATEGORICAL BRANCH          │   │
│  │                     │    │                                  │   │
│  │  SimpleImputer      │    │  SimpleImputer                   │   │
│  │  (median)           │    │  (most_frequent)                 │   │
│  │       ↓             │    │       ↓                          │   │
│  │  StandardScaler     │    │  OneHotEncoder                   │   │
│  │                     │    │  (handle_unknown='ignore')       │   │
│  └────────┬────────────┘    └───────────────┬──────────────────┘   │
│           │                                 │                       │
│           └──────────────┬──────────────────┘                      │
│                          ▼                                          │
│                   ColumnTransformer                                 │
│                          │                                          │
│                          ▼                                          │
│              ┌───────────────────────┐                             │
│              │      CLASSIFIER       │                             │
│              │  (5 models compared)  │                             │
│              └───────────────────────┘                             │
│                          │                                          │
│                          ▼                                          │
│                  ESTRATO prediction                                 │
│              (multiclass: 4 strata)                                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Validation strategy:** `StratifiedKFold(n_splits=10)` — preserves class distribution across all folds, critical for an imbalanced multiclass target.

**Hyperparameter tuning:** `RandomizedSearchCV(n_iter=30)` applied to both Random Forest and XGBoost with independent parameter spaces.

---

## 📁 Project Structure

```
Prediccion-de-estrato-TI/
├── ML_Proyecto2.ipynb                    # 📓 Full DS pipeline (EDA → modeling → evaluation)
├── tr_endutih_hogares_anual_2024.csv     # 📦 Source dataset (INEGI ENDUTIH 2024, public)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Execution

```bash
# 1. Clone the repository
git clone https://github.com/JuanDiego076/Prediccion-de-estrato-TI.git
cd Prediccion-de-estrato-TI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook ML_Proyecto2.ipynb
```

> **Google Colab users:** Update the `file_path` variable in the notebook to point to your Drive mount path.

---

## 🔑 Key Technical Decisions

**ColumnTransformer Pipeline (not a single encoder for all features):** The dataset mixes truly numeric variables (device counts, connection quantities) with binary/coded categoricals (presence/absence indicators, service type codes). Applying a single encoding strategy to all features would either distort numeric relationships or lose ordinality in categoricals. The `ColumnTransformer` handles each type with the appropriate transformer — `StandardScaler` for numerics, `OneHotEncoder` for categoricals — within a single `sklearn.Pipeline` object. This ensures zero data leakage between train and test folds during cross-validation.

**Heuristic for categorical detection:** Variables with ≤10 unique numeric values are reclassified as categorical at preprocessing time. This captures the many binary (1/2) and ordinal (1/2/3) columns common in INEGI survey microdata.

**Null value strategy:** Columns with >70% missing values are dropped before feature selection. Remaining nulls are handled within the pipeline imputers (median for numeric, most frequent for categorical) — ensuring the imputation statistics are learned only from training data.

**Bagging over XGBoost as the winner:** On this dataset, Bagging with Decision Tree base estimators outperforms gradient boosting. The likely reason: Bagging's variance reduction mechanism handles the combination of sparse categorical features (post-OHE) and moderate null rates better than AdaBoost/XGBoost's sequential correction approach.

---

## 📝 Lessons Learned

- AdaBoost's poor performance (~70%) confirmed that sequential boosting is fragile under class imbalance and noisy survey variables. Isolating that failure mode early redirected attention toward Random Forest / Bagging as the viable production candidates.
- `StratifiedKFold` is non-negotiable for socioeconomic data: ESTRATO distributions in INEGI surveys are not uniform, and standard KFold would produce unreliable fold compositions.
- `RandomizedSearchCV` over `GridSearchCV` at n_iter=30 covers the search space in ~1/10th the compute time — adequate for exploratory model selection at this scale.

---

## 📦 Dataset

**Source:** [ENDUTIH 2024 — INEGI](https://www.inegi.org.mx/programas/endutih/2024/)  
**Scope:** National survey on ICT availability and usage in Mexican households (2024 annual edition)  
**License:** Public domain — INEGI open data policy  
**Shape:** 58,080 households × ~100 variables (ICT indicators + household characteristics)

---

## 👤 Author

**Juan Diego Taborda Roldán**  
Telecommunications Engineer | Data Engineer & Data Scientist  
[LinkedIn](https://www.linkedin.com/in/juandiegotabordaroldan) · [GitHub](https://github.com/JuanDiego076) · [juandiego276@gmail.com](mailto:juandiego276@gmail.com)

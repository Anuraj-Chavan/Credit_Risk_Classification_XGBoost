# 🪙 Credit Risk Classification

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-%23EB4034.svg?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

**A multiclass credit risk classification system that predicts customer risk categories using statistical feature selection and ensemble machine learning.**

## 🧭 Project Overview

In the financial sector, accurately assessing the creditworthiness of customers is critical for reducing lending risk and supporting better approval decisions.

This project builds a machine learning classification system that analyzes demographic, financial, credit, and product-enquiry information to classify applicants into four credit risk categories: **P1, P2, P3, and P4**.

The project combines statistical analysis, feature selection, categorical encoding, and multiple machine learning models. The final **XGBoost** model is serialized and integrated into a Streamlit web application for real-time credit risk prediction.

## 🧩 Technical Architecture & Workflow

The system is organized into three main components to maintain a clear separation between data analysis, model training, and application inference.

### 1. Data Pipeline & Model Training (`Credit_Risk_Classification_FINAL.ipynb`)

The training phase applies statistical and machine learning techniques to prepare the data and build the classification models:

* **Data Preparation:**
    * Two source datasets, `case_study1.xlsx` and `case_study2.xlsx`, are cleaned and merged using `PROSPECTID`.
    * Invalid placeholder values such as `-99999` are handled during preprocessing.

* **Feature Selection:**
    * **Chi-Square Test:** Used to evaluate relationships between categorical features and the target variable.
    * **VIF Analysis:** Sequential Variance Inflation Factor analysis is performed on numerical variables to reduce multicollinearity.
    * **ANOVA:** Used to identify significant numerical features across the credit risk categories.

* **Feature Engineering:**
    * **Ordinal Encoding:** Education levels are converted into numerical values while preserving their intended order.
    * **One-Hot Encoding:** Categorical variables such as `MARITALSTATUS`, `GENDER`, `last_prod_enq2`, and `first_prod_enq2` are transformed into machine-readable binary features.

* **Model Selection:**
    * **Random Forest**
    * **Decision Tree**
    * **XGBoost**

    These models are evaluated and compared for multiclass credit risk classification.

* **Final Model:**
    * XGBoost is used as the final prediction engine.
    * The target classes are encoded as **P1 → 0, P2 → 1, P3 → 2, P4 → 3**.

### 2. Inference Engine (`app.py`)

The Streamlit application acts as the bridge between user-provided applicant information and the trained XGBoost model.

* **Model Loading:** Loads the serialized XGBoost model, label encoder, and feature-name structure.
* **Input Processing:** Converts user inputs into the same feature structure used during model training.
* **Categorical Encoding:** Recreates the one-hot encoded categorical representation expected by the model.
* **Education Mapping:** Applies the same ordinal education mapping used during training.
* **Prediction:** Passes the processed feature vector to XGBoost to generate the predicted credit risk category.
* **Probability Scoring:** Displays prediction probabilities for each risk category.

### 3. Interactive Dashboard (`app.py`)

* **Framework:** Built with **Streamlit** for interactive real-time inference.
* **User Experience:** Provides a structured form for entering applicant information.
* **Prediction Output:** Displays the predicted category along with probability scores for **P1, P2, P3, and P4**.
* **Model Information:** The sidebar provides information about the model and the number of features used for prediction.

## 📊 Key Features

* **Multiclass Risk Classification:** Predicts four credit risk categories — **P1, P2, P3, and P4**.
* **Statistical Feature Selection:** Uses Chi-Square, VIF, and ANOVA analysis to support feature selection.
* **Model Comparison:** Evaluates Decision Tree, Random Forest, and XGBoost models.
* **ROC-AUC Evaluation:** Uses multiclass Macro ROC-AUC and One-vs-Rest ROC curves to evaluate model performance.
* **Probability Scoring:** Provides class-wise prediction probabilities instead of only returning the predicted class.
* **Real-time Inference:** Streamlit enables users to enter applicant information and receive predictions interactively.
* **Reproducible Model Artifacts:** Stores the trained XGBoost model, label encoder, and feature names as serialized files.

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/credit-risk-classification.git
   cd credit-risk-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**
   ```bash
   streamlit run app.py
   ```

4. **Usage**
   * Enter the applicant's demographic, education, product-enquiry, and financial information.
   * Click **"Predict Credit Risk"**.
   * View the predicted credit risk category.
   * Review the probability assigned to each category.

## 🗂️ File Structure

```text
Credit-Risk-Classification/
│
├── Data/
│   ├── case_study1.xlsx
│   └── case_study2.xlsx
│
├── app.py
├── Credit_Risk_Classification_FINAL.ipynb
├── requirements.txt
│
├── xgboost_credit_risk_model.pkl
├── xgboost_feature_names.pkl
└── xgboost_label_encoder.pkl
```

### File Description

* `Data/case_study1.xlsx`: First source dataset used for credit risk analysis.
* `Data/case_study2.xlsx`: Second source dataset used for credit risk analysis.
* `Credit_Risk_Classification_FINAL.ipynb`: Complete notebook containing data preprocessing, statistical analysis, feature selection, model training, model comparison, and ROC-AUC evaluation.
* `app.py`: Streamlit application for real-time credit risk prediction.
* `requirements.txt`: Pinned Python dependencies required to reproduce the project environment.
* `xgboost_credit_risk_model.pkl`: Serialized trained XGBoost classification model.
* `xgboost_feature_names.pkl`: Saved feature names and ordering used by the trained model.
* `xgboost_label_encoder.pkl`: Serialized label encoder used to convert between encoded predictions and P1–P4 risk categories.

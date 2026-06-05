# Telecom Customer Churn Prediction

## Project Overview

Customer churn is one of the most critical challenges faced by telecommunications companies. Retaining existing customers is often more cost-effective than acquiring new ones. This project aims to analyze customer behavior and predict churn risk using Machine Learning techniques.

The project follows the complete Data Science lifecycle, including data cleaning, exploratory data analysis, feature engineering, feature selection, model building, hyperparameter tuning, evaluation, and deployment using Streamlit.

---

## Business Problem

Telecommunication companies lose significant revenue when customers discontinue their services. By identifying customers who are likely to churn, businesses can take proactive actions to improve customer satisfaction and retention.

### Objectives

* Analyze customer behavior and engagement.
* Identify factors influencing customer churn.
* Build predictive machine learning models.
* Compare multiple algorithms.
* Deploy the best model through a Streamlit web application.

---

## Dataset Information

The dataset contains customer demographic information, engagement metrics, transaction behavior, complaint history, feedback, membership information, and churn status.

### Target Variable

* `churn_risk_score`

### Key Features

* Age
* Gender
* Membership Category
* Region Category
* Average Time Spent
* Average Transaction Value
* Points in Wallet
* Days Since Last Login
* Login Frequency
* Complaint Status
* Feedback
* Internet Option
* Referral Status

---

## Project Workflow

### 1. Data Cleaning

The following preprocessing steps were performed:

* Removed duplicate records
* Handled missing values
* Fixed invalid values
* Removed inconsistent records
* Processed date features
* Standardized categorical variables
* Renamed columns for readability

### 2. Exploratory Data Analysis (EDA)

Business questions investigated:

1. Does membership category affect churn?
2. Do customer complaints increase churn risk?
3. Does customer engagement affect churn?
4. Does transaction value influence churn?
5. Does customer feedback impact churn behavior?
6. Which factors are most correlated with churn?

Visualizations used:

* Count Plots
* Histograms
* Box Plots
* Heatmaps
* Correlation Analysis
* Bar Charts

### 3. Feature Engineering

New features created:

* Age Group
* High Value Customer
* Active Customer

### 4. Feature Selection

Feature importance and correlation analysis were used to identify the most relevant predictors.

### 5. Machine Learning Models

Three classification algorithms were trained and evaluated:

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier

### 6. Hyperparameter Tuning

GridSearchCV was used to optimize the Random Forest model and improve predictive performance.

### 7. Evaluation Metrics

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### 8. Deployment

The final model was deployed using Streamlit, allowing users to input customer information and receive churn predictions in real time.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib
* Streamlit

---

## Project Structure

```text
Telecomunication_Churn_Perdiction_Ml_Model/
│
├── churn.csv
├── project.ipynb
├── app.py
├── churn_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
```

---

## Results

The machine learning models were compared using multiple evaluation metrics. The best-performing model was selected and deployed for real-time churn prediction.

Key findings:

* Customer engagement strongly impacts churn behavior.
* Membership category is a significant predictor of churn.
* Customer complaints and negative feedback are associated with higher churn risk.
* Spending behavior and wallet points contribute to customer retention.

---

## Author

 

Saif Eldeen Wesam Elsayed 

Epsilon AI Data Science Final Project

2026

# Customer Churn Prediction & Analysis

## 1. Introduction:
This project is about building a machine learning model to predict customer churn in a telecom firm. The project is built on 4 main phases: 
1. The first phase: utilizes data analysis techniques such as data cleaning and EDA to understand the data and deliver insights about any hidden patterns within data. 

2. The second phase: based on these insights we use feature transformation and feature engineering to transform the data in a form that would be most appropriate for machine learning. 

3. The third phase: this is the machine learning model development, where we train and test different machine learning models (classifiers) with the obtained - cleaned and feature engineered - data, namely: 
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoosting
- Support Vector Machines

Each of those models is trained and testing twice:
1. Basic model algorithm
2. Model algorithm with Grid Search Cross-Validation

Alongside model training and testing, we perform MLOps for tracking the model training and testing experiments using MLFlow.

4. The fourth phase: Based on the results of the classifiers, we choose the one with the best results for deployment, which we use the streamlit.io platform to do so.

## 2. Repository Folder Structure:

### 2.1 Data Folder: 
This folder contains several datasets


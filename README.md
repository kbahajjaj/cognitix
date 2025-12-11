# Customer Churn Prediction & Analysis

## 1. Introduction:
This project was made as a graduation requirement from the DEPI (Digital Egypt Pioneers Initiative). It is about building a machine learning model to predict customer churn in a telecom firm. The project is built on 4 main phases: 
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
This folder contains several datasets:
1. telecom_customer_churn.csv --> Original raw data which was downloaded from the following url:
   https://www.kaggle.com/code/zakriasaad1/customer-churn-prediction-on-telecom-dataset/data
2. telecom_customer_churn_clean.csv --> the preprocessed data (cleaned and feature engineered) for machine learning models.
3. telecom_data_dictionary.csv --> the file that describes the meaning of the columns of the raw dataset.
4. 3 datasets for testing on the deployed model (streamlit webapp) --> these 3 datasets has identical 34 columns, but only differ in the number of entries (5, 10, and 15 rows). This is to use the data to get predictions for different number of entries each time.
5. Also you can use the datasets from 1 to 3 on the webapp to see how does the deployed model handles erroneous inputs.

### 2.2 Docs Folder:
This folder contains the documents that were required by the DEPI and MCIT (Ministry of Communications & Information Technology - Egypt) as a part of the project:
1. Customer Churn Prediction & Analysis - Slides.pdf: Project presentation slides
2. Customer Churn Prediction & Analysis.pdf: Project documentation

### 2.3 MLFlow Screenshots Folder:
These are the screenshots that were taken from the MLFlow dashboard which displays the results of the training/testing experiments.

### 2.4 Notebooks Folder:
This folder has 7 Jupyter Notebooks:
1. Data preprocessing
2. Logistic regression model
3. Random forest model
4. Gradient Boosting Model
5. XGBoost Model
6. SVM model
7. All models in a single notebook

### 2.5 Streamlit Folder:
This folder contains the required files for running the webapp: address = https://cognitix.streamlit.app:
1. requirements.txt file
2. the model with the highest results (random forest with grid search) saved in a .pkl file
3. the logo for the webapp
4. streamlit_app.py: the python file required to run the webapp
   


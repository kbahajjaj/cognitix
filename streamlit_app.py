import streamlit as st
import pandas as pd
import joblib
import os

# Define the relative path to your model file
DIRPATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(DIRPATH, "rf_gscv_model.pkl")
SCALER_PATH = os.path.join(DIRPATH, "rf_gscv_model.pkl")


# Load the trained model
@st.cache_resource
def load_object(path):
    with open(path, 'rb') as f:
        loaded_object = joblib.load(f)
    return loaded_object

model = load_object(MODEL_PATH)
scaler = load_object(SCALER_PATH)

# --- Streamlit App Interface ---
st.logo(image="logo.png", size="medium")
st.header("Customer Churn Analysis & Prediction")
st.write("Enter input features to get a prediction:")

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file, encoding='cp1252')

    # Display the DataFrame in the Streamlit app
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Ensure there's enough data for a meaningful model (more than 1 column)
    if df.shape[1] == 34:
        input_df = df

        # Make and display prediction when a button is clicked
        if st.button("Predict"):
            st.subheader("Prediction")
            for i in range(len(input_df)):
                prediction = model.predict(input_df.iloc[[i]])
                if prediction == 1:
                    customer_status = 'Churned'
                else:
                    customer_status = 'Stayed'
                st.success(f"The predicted customer state for entry {i} --> {customer_status}")

    else:
        st.warning(f"Warning: Your uploaded CSV file has {df.shape[1]} columns.\n\
                   It must have 34 columns/features for the Random Forest model to work.")
        st.warning("Required: Upload a CSV file with 34 columns.")
else:
    st.info("Please upload a CSV file to proceed.")

# # Make and display prediction when a button is clicked
# if st.button("Predict"):
#     st.subheader("Prediction")
#     for i in range(len(input_df)):
#         prediction = model.predict(input_df.iloc[[i]])
#         if prediction == 1:
#             customer_status = 'Churned'
#         else:
#             customer_status = 'Stayed'
#         st.success(f"The predicted customer state for entry {i} is: {customer_status}")

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the pre-trained models
kmeans_model = joblib.load('kmeans_model.pkl')
classifier_model = joblib.load('classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit App Layout
st.title("Customer Segmentation and Prediction App")

st.markdown("""
This app uses **K-means clustering** to segment customers and a **Random Forest Classifier** 
to predict the cluster for new customer data based on age and annual income.
""")

# User Inputs
age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Enter Annual Income", min_value=1000, max_value=200000, value=50000)

# Create feature array
input_data = np.array([[age, annual_income]])

# Scale the input data for K-means
scaled_input = scaler.transform(input_data)

# Predict cluster using K-means
kmeans_cluster = kmeans_model.predict(scaled_input)
st.write(f"**K-means Predicted Cluster:** {kmeans_cluster[0]}")

# Predict cluster using Random Forest Classifier
rf_cluster = classifier_model.predict(input_data)
st.write(f"**Random Forest Predicted Cluster:** {rf_cluster[0]}")

# Show cluster centers
if st.checkbox("Show Cluster Centers (K-means)"):
    cluster_centers = scaler.inverse_transform(kmeans_model.cluster_centers_)
    centers_df = pd.DataFrame(cluster_centers, columns=['Age', 'Annual Income'])
    st.write("**Cluster Centers:**")
    st.write(centers_df)

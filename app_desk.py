import streamlit as st
import boto3
import json
import pandas as pd
import numpy as np
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sagemaker.session import Session
from io import StringIO
import requests
import joblib
from io import BytesIO

aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
aws_region = st.secrets["AWS_DEFAULT_REGION"]

# Read CSV from S3
@st.cache_data
def read_csv_from_url(url):
    return pd.read_csv(url)


@st.cache_resource
def load_scaler(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        scaler = joblib.load(BytesIO(response.content))
        return scaler
    except Exception as e:
        st.error(f"Failed to load scaler: {e}")
        return None


@st.cache_resource
def load_pca(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()  
        pca = joblib.load(BytesIO(response.content))  
        return pca
    except Exception as e:
        st.error(f"Failed to load PCA model: {e}")
        return None

# AWS SageMaker Endpoint
ENDPOINT_NAME = "CarRecommendationEndpointMoeThree3"

boto3.setup_default_session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

predictor = Predictor(
    endpoint_name=ENDPOINT_NAME, 
    serializer=CSVSerializer(),
    sagemaker_session=Session(boto3.Session(region_name=aws_region))
)

# Load encoders
category_encoder = LabelEncoder()
gearbox_encoder = LabelEncoder()
fueltype_encoder = LabelEncoder()
category_encoder.fit(["OffRoad", "Van", "Limousine", "Estate Car", "Small car", "Sport Car"])
gearbox_encoder.fit(["Manual", "Automatic"])
fueltype_encoder.fit(["Petrol", "Diesel", "Electric", "Hybrid"])

# Initialize StandardScaler
scaler_url = "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/model/scaler.pkl"
scaler = load_scaler(scaler_url)


# Load PCA model from S3
pca_url = "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/model/pca.pkl"
pca = load_pca(pca_url)

# Streamlit UI
st.title("Car Recommendation System")

category = st.selectbox("Category", ["OffRoad", "Van", "Limousine", "Estate Car", "Small car", "Sport Car"])
gearbox = st.selectbox("Gearbox", ["Manual", "Automatic"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
price = st.number_input("Price ($)", min_value=1000, max_value=200000, step=500)
mileage = st.number_input("Mileage (km)", min_value=0, max_value=300000, step=5000)
performance = st.number_input("Performance (HP)", min_value=50, max_value=1000, step=10)
first_reg = st.number_input("First Registration Year", min_value=1990, max_value=2025, step=1)

if st.button("Get Recommendation"):
    url = "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/WithClusterTest/predictions_with_clusters.csv"
    ads_df = read_csv_from_url(url)
    
    # Fit scaler on dataset
    numerical_features = ["Price", "Mileage", "Performance", "FirstReg"]
   
    
    new_ad = pd.DataFrame({
        "Category": [category],
        "FirstReg": [first_reg],
        "Gearbox": [gearbox],
        "Price": [price],
        "FuelTyp": [fuel_type],
        "Mileage": [mileage],
        "Performance": [performance]
    })
    
    new_ad["Category"] = category_encoder.transform(new_ad["Category"])
    new_ad["Gearbox"] = gearbox_encoder.transform(new_ad["Gearbox"])
    new_ad["FuelTyp"] = fueltype_encoder.transform(new_ad["FuelTyp"])
    
    new_ad[numerical_features] = scaler.transform(new_ad[numerical_features])

     # Apply PCA before making predictions
    if pca is not None:
        new_ad_pca = pca.transform(new_ad)
    else:
        st.error("❌ PCA model not loaded.")
        st.stop()

    # Convert to list for SageMaker inference
    data_to_predict = new_ad_pca.tolist()
    
    # data_to_predict = new_ad.values.tolist()
    response = predictor.predict(data_to_predict)
    response = response.decode("utf-8") if isinstance(response, bytes) else response
    
    try:
        predictions = json.loads(response)
        predicted_cluster = predictions["predictions"][0] if isinstance(predictions, dict) else predictions[0]
    except json.JSONDecodeError:
        predicted_cluster = int(response.strip())
    
    st.success(f"Recommended Cluster: {predicted_cluster}")
    
    if "cluster" in ads_df.columns:
        similar_ads = ads_df[ads_df["cluster"] == predicted_cluster]
        similar_ads = similar_ads.sort_values(by=["Price", "Mileage", "Performance"], ascending=True)
        st.success(f"✅ Found {len(similar_ads)} similar ads in cluster {predicted_cluster}")
        st.dataframe(similar_ads.head(10))
    else:
        st.error("❌ The dataset is missing the 'cluster' column.")

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

def read_csv_from_s3(bucket_name, file_key):
    """Read a CSV file from S3 into a Pandas DataFrame."""
    boto3.setup_default_session(
    aws_access_key_id="AKIA6CN7Q7IF3RFLP37I",
    aws_secret_access_key="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",
    region_name="us-east-1")
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response["Body"].read().decode("utf-8")
    return pd.read_csv(StringIO(csv_content))

# AWS SageMaker Endpoint
ENDPOINT_NAME = "CarRecommendationEndpointMoeThree3"
# Manually set AWS credentials
boto3.setup_default_session(
    aws_access_key_id="AKIA6CN7Q7IF3RFLP37I",
    aws_secret_access_key="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",
    region_name="us-east-1"
)

# Create a SageMaker Predictor
predictor = Predictor(endpoint_name=ENDPOINT_NAME, serializer=CSVSerializer(),sagemaker_session=Session(boto3.Session(region_name="us-east-1")))

# Load encoders (same as training)
category_encoder = LabelEncoder()
gearbox_encoder = LabelEncoder()
fueltype_encoder = LabelEncoder()

category_encoder.fit(["SUV", "Sedan", "Convertible"])
gearbox_encoder.fit(["Manual", "Automatic"])
fueltype_encoder.fit(["Petrol", "Diesel", "Electric"])

# StandardScaler (Use the same mean & scale as in training)
scaler = StandardScaler()
scaler.mean_ = np.array([20000, 60000, 120, 2015])
scaler.scale_ = np.array([5000, 20000, 50, 5])

# Streamlit UI
st.title("Car Recommendation System")

# Input fields
category = st.selectbox("Category", ["SUV", "Sedan", "Convertible"])
gearbox = st.selectbox("Gearbox", ["Manual", "Automatic"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"])
price = st.number_input("Price ($)", min_value=1000, max_value=200000, step=500)
mileage = st.number_input("Mileage (km)", min_value=0, max_value=300000, step=5000)
performance = st.number_input("Performance (HP)", min_value=50, max_value=1000, step=10)
first_reg = st.number_input("First Registration Year", min_value=1990, max_value=2025, step=1)

# Submit button
if st.button("Get Recommendation"):
    bucket_name = "car-recommendation-raed"
    file_key = "WithClusterTest/predictions_with_clusters.csv"
    # Create DataFrame
    new_ad = pd.DataFrame({
        "Category": [category],
        "FirstReg": [first_reg],
        "Gearbox": [gearbox],
        "Price": [price],
        "FuelTyp": [fuel_type],
        "Mileage": [mileage],
        "Performance": [performance]
    })

    # Encode categorical values
    new_ad["Category"] = category_encoder.transform(new_ad["Category"])
    new_ad["Gearbox"] = gearbox_encoder.transform(new_ad["Gearbox"])
    new_ad["FuelTyp"] = fueltype_encoder.transform(new_ad["FuelTyp"])

    # Standardize numerical values
    numerical_features = ["Price", "Mileage", "Performance", "FirstReg"]
    new_ad[numerical_features] = scaler.transform(new_ad[numerical_features])

    # Convert to list and send to SageMaker
    data_to_predict = new_ad.values.tolist()
    response = predictor.predict(data_to_predict)

    # Decode response
    response = response.decode("utf-8") if isinstance(response, bytes) else response

    try:
        # Try parsing JSON response
        predictions = json.loads(response)
        if isinstance(predictions, dict) and "predictions" in predictions:
            predicted_cluster = predictions["predictions"][0]  # Dictionary format
        elif isinstance(predictions, list) and len(predictions) > 0:
            predicted_cluster = predictions[0]  # List format
        else:
            st.error("Unexpected response format from SageMaker")
            raise ValueError("Unknown response format")
    except json.JSONDecodeError:
        predicted_cluster = int(response.strip())

    # Display result
    st.success(f"Recommended Cluster: {predicted_cluster}")

    try:
        ads_df = read_csv_from_s3(bucket_name, file_key)

        # Ensure dataset has a "cluster" column
        if "cluster" not in ads_df.columns:
            st.error("❌ The dataset is missing the 'cluster' column.")
        else:
            # Simulating a predicted cluster (replace this with actual model prediction)
               
            # Filter ads in the same cluster
            similar_ads = ads_df[ads_df["cluster"] == predicted_cluster]

            # Sort similar ads by Price, Mileage, and Performance
            similar_ads = similar_ads.sort_values(by=["Price", "Mileage", "Performance"], ascending=True)

            # Display results
            st.success(f"✅ Found {len(similar_ads)} similar ads in cluster {predicted_cluster}")
            st.dataframe(similar_ads.head(10))  # Show top 10 ads

    except Exception as e:
        st.error(f"⚠ Error loading data: {e}")
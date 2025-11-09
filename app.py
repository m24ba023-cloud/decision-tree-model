import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load trained model
loaded_dt_model = joblib.load('decision_tree_model(1).joblib')

# Define feature names
feature_names = [
    'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition', 'Delivery_Slot',
    'Driver_Experience', 'Num_Stops', 'Vehicle_Age', 'Road_Condition_Score',
    'Package_Weight', 'Fuel_Efficiency', 'Warehouse_Processing_Time'
]

# Prediction function
def predict_delivery_delay(delivery_distance, traffic_congestion, weather_condition, delivery_slot,
                           driver_experience, num_stops, vehicle_age, road_condition_score,
                           package_weight, fuel_efficiency, warehouse_processing_time):
    input_features = np.array([[
        delivery_distance, traffic_congestion, weather_condition, delivery_slot,
        driver_experience, num_stops, vehicle_age, road_condition_score,
        package_weight, fuel_efficiency, warehouse_processing_time
    ]])

    input_df = pd.DataFrame(input_features, columns=feature_names)
    prediction = loaded_dt_model.predict(input_df)
    return "Delayed" if prediction[0] == 1 else "On Time"


# Streamlit UI
st.set_page_config(page_title="Delivery Delay Prediction App", layout="centered")
st.title("Delivery Delay Prediction App")
st.write("Enter the details below to predict if a delivery will be delayed.")

st.header("Delivery Parameters")
col1, col2 = st.columns(2)

with col1:
    delivery_distance = st.slider('Delivery Distance (km)', 0.0, 100.0, 25.0)
    traffic_congestion = st.selectbox('Traffic Congestion Level', [1, 2, 3, 4, 5], index=2)
    weather_condition = st.selectbox('Weather Condition', [1, 2, 3, 4, 5], index=2)
    delivery_slot = st.selectbox('Delivery Time Slot', [1, 2, 3], index=1)
    driver_experience = st.slider('Driver Experience (Years)', 0, 20, 5)

with col2:
    num_stops = st.slider('Number of Stops', 1, 10, 3)
    vehicle_age = st.slider('Vehicle Age (Years)', 0, 15, 3)
    road_condition_score = st.selectbox('Road Condition Score', [1, 2, 3, 4, 5], index=2)
    package_weight = st.slider('Package Weight (kg)', 0.0, 50.0, 10.0)
    fuel_efficiency = st.slider('Fuel Efficiency (km/L)', 5.0, 20.0, 12.0)
    warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', 0, 90, 30)

if st.button('Predict Delivery Delay Status'):
    result = predict_delivery_delay(
        delivery_distance, traffic_congestion, weather_condition,
        delivery_slot, driver_experience, num_stops, vehicle_age,
        road_condition_score, package_weight, fuel_efficiency,
        warehouse_processing_time
    )

    if result == "Delayed":
        st.error(f"The predicted delivery status is: {result}")
    else:
        st.success(f"The predicted delivery status is: {result}")

st.info("A prediction of 'Delayed' means the delivery is likely to be late.")

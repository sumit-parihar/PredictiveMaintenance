import streamlit as st          # For the GUI
import pandas as pd             # Optional: for logging predictions
import joblib                   # To load the trained ML model
import os                       # To check if file path exist or not

# Load the trained model from models folder
model = joblib.load("models/predictive_model.pkl")

# Set up App interface
st.title("Predictive Maintenance App")
st.write("Enter machine details to predict if maintenance is required")

# User input fields
air_temp = st.number_input("Air Temperature [K]", min_value=200.0, max_value=500.0, value=300.0, step=1.0)
process_temp = st.number_input("Process Temperature [K]", min_value=200.0, max_value=500.0, value=300.0, step=1.0)
rot_speed = st.number_input("Rotational Speed [rpm]", min_value=0, max_value=5000, value=1500)
torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0)
tool_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=500, value=10)
product_quality = st.selectbox("Product Quality (L=0, M=1, H=2)", [0, 1, 2])

# Add Prediction Button
if st.button("Predict Maintenance"):

    # Prepare input for model
    input_data = [[air_temp, process_temp, rot_speed, torque, tool_wear, product_quality]]

    # Make Prediction
    prediction = model.predict(input_data)

    # Convert numeric prediction to readable text
    result = "Yes" if prediction == 1 else "No"

    # Display Result
    st.success(f"Maintenance Prediction: {result}")

    # Log the prediction
    log_df = pd.DataFrame([[air_temp, process_temp, rot_speed, torque, tool_wear, product_quality, result]],
                      columns=['AirTemp', 'ProcessTemp', 'RotSpeed', 'Torque', 'ToolWear', 'ProductQuality', 'Prediction'])

    # Set file path
    file_path = "data/prediction_log.csv"

    # Determine starting index
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        start_index = len(existing_df) + 1
    else:
        start_index = 1

    # Set index for new row(s)
    log_df.index = range(start_index, start_index + len(log_df))

    # Append to CSV
    log_df.to_csv("data/prediction_log.csv", mode = 'a', index = True, index_label='Serial No.', header = not os.path.exists(file_path))
import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import os

logs_dir = "/content/drive/MyDrive/DaoTaoTinChi/TIN4653_HocMayVoiPython/BaiGiang/2025/Streamlit_MLFlow_GDrive/mlflow_logs"

logs_uri = "file://" + logs_dir

# Đặt đường dẫn đến mô hình đã lưu trên Google Drive
MLFLOW_TRACKING_URI = logs_uri

# Load mô hình từ Google Drive
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_ID = "685042325580286509"
RUN_ID = "ef97695d49b24e56926b490c531ef15c"
model_path = os.path.join(logs_dir, EXPERIMENT_ID, RUN_ID, "artifacts", "linear_regression_model")
model = mlflow.sklearn.load_model(model_path)

# Giao diện Streamlit
st.title("Linear Regression Model Deployment")
st.write("Nhập giá trị X để dự đoán Y")

# Nhập giá trị X từ người dùng
input_value = st.number_input("Nhập giá trị X:", min_value=0.0, max_value=10.0, value=5.0)

# Dự đoán giá trị Y
if st.button("Dự đoán"):
    prediction = model.predict(np.array([[input_value]]))
    st.write(f"Kết quả dự đoán: Y = {prediction[0][0]:.2f}")


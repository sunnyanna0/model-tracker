from pymongo import MongoClient
from dotenv import load_dotenv
import os
import pandas as pd

# .env 파일 로드
load_dotenv()

# 환경 변수에서 URI 불러오기
mongodb_uri = os.getenv("MONGODB_URI")

# MongoDB 클라이언트 설정
client = MongoClient(mongodb_uri)
db = client["model_logs_db"]
trainings = db["trainings"]
epochs = db["epochs"]

#---모델 리스트 / 선택
import streamlit as st

# 모든 run_id 목록 표시
run_ids = [doc["run_id"] for doc in trainings.find()]

#---선택된 run_id의 메타 정보 표시
selected_run = st.selectbox("모델 실행(run_id) 선택", run_ids)
run_info = trainings.find_one({"run_id": selected_run})
st.write("모델:", run_info["model_name"])
st.write("버전:", run_info["version"])
st.write("최종 Train Accuracy:", run_info["train_accuracy"])
st.write("최종 Val Accuracy:", run_info["val_accuracy"])
st.write("Test Accuracy:", run_info["test_accuracy"])
st.write("학습 소요 시간:", f"{run_info['duration_minutes']:.2f}분")

#--- 에폭별 정확도 & 손실 시각화
import matplotlib.pyplot as plt

epoch_logs = list(epochs.find({"run_id": selected_run}).sort("epoch", 1))
df = pd.DataFrame(epoch_logs)

st.subheader("📈 정확도 / 손실 추이")

# 정확도 그래프
fig1, ax1 = plt.subplots()
ax1.plot(df["epoch"], df["train_acc"], label="Train Accuracy", color="blue")
ax1.plot(df["epoch"], df["val_acc"], label="Val Accuracy", color="orange")
ax1.set_title("Train vs Validation Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
st.pyplot(fig1)

# 손실 그래프
fig2, ax2 = plt.subplots()
ax2.plot(df["epoch"], df["train_loss"], label="Train Loss", color="blue")
ax2.plot(df["epoch"], df["val_loss"], label="Val Loss", color="orange")
ax2.set_title("Train vs Validation Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
st.pyplot(fig2)

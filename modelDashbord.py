from pymongo import MongoClient
from dotenv import load_dotenv
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# .env 파일 로드
load_dotenv()

# 환경 변수에서 URI 불러오기
mongodb_uri = os.getenv("MONGODB_URI")

# MongoDB 클라이언트 설정
client = MongoClient(mongodb_uri)
db = client["model_logs_db"]
trainings = db["trainings"]
epochs = db["epochs"]

st.title("🧠 모델 학습 로그 대시보드")

# --- 메뉴 선택 ---
menu = st.sidebar.radio("📌 메뉴 선택", ["모델 상세 보기", "📊 모델 비교 보기"])

if menu == "모델 상세 보기":
    run_ids = [doc["run_id"] for doc in trainings.find({}, {"run_id": 1})]
    selected_run = st.selectbox("모델 실행(run_id) 선택", run_ids)
    run_info = trainings.find_one({"run_id": selected_run})

    st.subheader("📌 모델 정보")
    st.markdown(f"**모델 이름:** `{run_info['model_name']}`")
    st.markdown(f"**버전:** `{run_info['version']}`")
    st.markdown(f"**클래스 수:** `{run_info['num_classes']}`")
    st.markdown(f"**Dropout:** `{run_info.get('dropout', 'None')}`")
    st.markdown(f"**Feature Extraction:** `{run_info.get('feature_extraction', False)}`")
    st.markdown(f"**Batch Size:** `{run_info['batch_size']}`")
    st.markdown(f"**Early Stopped:** `{run_info.get('early_stopped', False)}`")

    st.subheader("📊 최종 성능")
    st.metric("Train Accuracy", f"{run_info['train_accuracy']:.4f}")
    st.metric("Val Accuracy", f"{run_info['val_accuracy']:.4f}")
    st.metric("Test Accuracy", f"{run_info['test_accuracy']:.4f}")
    st.metric("Best Val Loss", f"{run_info['best_val_loss']:.4f}")

    st.markdown(f"**학습 소요 시간:** `{run_info['duration_minutes']:.2f}분`")
    st.markdown(f"**시작:** `{run_info['start_time']}`\n**종료:** `{run_info['end_time']}`")

    st.subheader("📈 Epoch별 추이 시각화")
    epoch_logs = list(epochs.find({"run_id": selected_run}).sort("epoch", 1))
    df = pd.DataFrame(epoch_logs)

    fig1, ax1 = plt.subplots()
    ax1.plot(df["epoch"], df["train_acc"], label="Train Accuracy", color="blue")
    ax1.plot(df["epoch"], df["val_acc"], label="Val Accuracy", color="orange")
    ax1.set_title("Train vs Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(df["epoch"], df["train_loss"], label="Train Loss", color="blue")
    ax2.plot(df["epoch"], df["val_loss"], label="Val Loss", color="orange")
    ax2.set_title("Train vs Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    st.pyplot(fig2)

    if "epoch_time_sec" in df.columns:
        st.subheader("⏱️ 에폭별 소요 시간")
        fig3, ax3 = plt.subplots()
        ax3.plot(df["epoch"], df["epoch_time_sec"], color="green")
        ax3.set_title("Epoch Time")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Time (sec)")
        st.pyplot(fig3)

    with st.expander("📄 원본 에폭 데이터 보기"):
        st.dataframe(df)

elif menu == "📊 모델 비교 보기":
    st.subheader("📊 전체 모델 성능 비교")
    all_runs = list(trainings.find())
    df_all = pd.DataFrame(all_runs)

    # 필터 조건 입력
    val_acc_threshold = st.slider("Val Accuracy ≥", min_value=0.0, max_value=1.0, step=0.01, value=0.0)
    filtered_df = df_all[df_all["val_accuracy"] >= val_acc_threshold]

    # 정렬
    sort_by = st.selectbox("정렬 기준", ["val_accuracy", "test_accuracy", "duration_minutes"], index=0)
    ascending = st.checkbox("오름차순 정렬", value=False)
    sorted_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

    # 보기 좋은 열만 선택
    display_cols = ["run_id", "model_name", "version", "val_accuracy", "test_accuracy", "train_accuracy", "best_val_loss", "early_stopped", "epochs_run", "duration_minutes"]
    st.dataframe(sorted_df[display_cols].reset_index(drop=True))

    import pandas as pd

#표 기반 성능 비교 (모델 vs 성능)
all_models = list(trainings.find({}, {
    "_id": 0,
    "model_name": 1,
    "version": 1,
    "train_accuracy": 1,
    "val_accuracy": 1,
    "test_accuracy": 1,
    "batch_size": 1,
    "dropout": 1,
    "learning_rate": 1
}))

df_models = pd.DataFrame(all_models)
st.subheader("📋 모델별 성능 비교")
st.dataframe(df_models)

import matplotlib.pyplot as plt

# 모델별 막대 그래프
st.subheader("📊 Train / Test Accuracy 비교")

fig, ax = plt.subplots(figsize=(8, 5))
model_names = df_models["model_name"] + " (" + df_models["version"] + ")"
x = range(len(model_names))

train_acc = df_models["train_accuracy"]
test_acc = df_models["test_accuracy"]

bar_width = 0.35
ax.bar(x, train_acc, width=bar_width, label='Train', color='mediumpurple')
ax.bar([i + bar_width for i in x], test_acc, width=bar_width, label='Test', color='midnightblue')

# 라벨 추가
for i in x:
    ax.text(i, train_acc[i] + 0.01, f"{train_acc[i]:.2f}", ha='center', fontsize=8)
    ax.text(i + bar_width, test_acc[i] + 0.01, f"{test_acc[i]:.2f}", ha='center', fontsize=8)

ax.set_xticks([i + bar_width / 2 for i in x])
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_ylabel("Accuracy")
ax.set_title("Train and Test Accuracy per Model")
ax.legend()
st.pyplot(fig)

st.subheader("🎯 하이퍼파라미터별 정확도")

# 선택 옵션: batch size, dropout, lr 등
param = st.selectbox("하이퍼파라미터 선택", ["batch_size", "dropout", "learning_rate"])
fig2, ax2 = plt.subplots()
df_models.sort_values(param, inplace=True)

ax2.plot(df_models[param], df_models["test_accuracy"], marker="o", label="Test Accuracy")
ax2.plot(df_models[param], df_models["train_accuracy"], marker="x", label="Train Accuracy")
ax2.set_xlabel(param)
ax2.set_ylabel("Accuracy")
ax2.set_title(f"Accuracy vs {param}")
ax2.legend()
st.pyplot(fig2)


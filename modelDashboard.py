from pymongo import MongoClient
from dotenv import load_dotenv
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime


# .env 파일 로드
load_dotenv()

# 환경 변수에서 URI 불러오기
mongodb_uri = os.getenv("MONGODB_URI")

# MongoDB 클라이언트 설정
client = MongoClient(mongodb_uri)
db = client["model_logs_db"]
trainings = db["trainings"]
epochs = db["epochs"]

st.title("💭 Model-tracker Dashboard")

# --- 메뉴 선택 ---
menu = st.sidebar.radio("📌 메뉴 선택", ["모델 상세 보기", "모델 비교 보기", "시간 필터", "학습 상태 실시간 모니터링"])


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

    # st.subheader("📈 Epoch별 추이 시각화")
    # epoch_logs = list(epochs.find({"run_id": selected_run}).sort("epoch", 1))
    # df = pd.DataFrame(epoch_logs)

    # fig1, ax1 = plt.subplots()
    # ax1.plot(df["epoch"], df["train_acc"], label="Train Accuracy", color="blue")
    # ax1.plot(df["epoch"], df["val_acc"], label="Val Accuracy", color="orange")
    # ax1.set_title("Train vs Validation Accuracy")
    # ax1.set_xlabel("Epoch")
    # ax1.set_ylabel("Accuracy")
    # ax1.legend()
    # st.pyplot(fig1)

    # fig2, ax2 = plt.subplots()
    # ax2.plot(df["epoch"], df["train_loss"], label="Train Loss", color="blue")
    # ax2.plot(df["epoch"], df["val_loss"], label="Val Loss", color="orange")
    # ax2.set_title("Train vs Validation Loss")
    # ax2.set_xlabel("Epoch")
    # ax2.set_ylabel("Loss")
    # ax2.legend()
    # st.pyplot(fig2)

    # if "epoch_time_sec" in df.columns:
    #     st.subheader("⏱️ 에폭별 소요 시간")
    #     fig3, ax3 = plt.subplots()
    #     ax3.plot(df["epoch"], df["epoch_time_sec"], color="green")
    #     ax3.set_title("Epoch Time")
    #     ax3.set_xlabel("Epoch")
    #     ax3.set_ylabel("Time (sec)")
    #     st.pyplot(fig3)

    # with st.expander("📄 원본 에폭 데이터 보기"):
    #     st.dataframe(df)

elif menu == "모델 비교 보기":
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

elif menu == "시간 필터":
    # --- 날짜 필터
    st.sidebar.header("📆 학습 시각 필터")
    start_date = st.sidebar.date_input("시작 날짜", datetime(2025, 3, 1).date())
    end_date = st.sidebar.date_input("종료 날짜", datetime.now().date())

    # MongoDB에 날짜 쿼리 적용
    filtered = list(trainings.find({
        "start_time": {
            "$gte": datetime.combine(start_date, datetime.min.time()),
            "$lte": datetime.combine(end_date, datetime.max.time())
        }
    }))

    if not filtered:
        st.warning("해당 기간에 학습된 모델이 없습니다.")
        st.stop()

    # --- 필터된 run_id 선택
    run_ids = [doc["run_id"] for doc in filtered]
    selected_run = st.selectbox("🔎 실행(run_id) 선택", run_ids)

    # --- 선택된 모델 정보 표시
    run_info = trainings.find_one({"run_id": selected_run})

    st.subheader("📌 모델 정보")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- **모델명:** `{run_info['model_name']}`")
        st.markdown(f"- **버전:** `{run_info['version']}`")
        st.markdown(f"- **Dropout:** `{run_info.get('dropout', 'None')}`")
        st.markdown(f"- **Batch Size:** `{run_info['batch_size']}`")
    with col2:
        st.markdown(f"- **LR:** `{run_info.get('learning_rate', 'Unknown')}`")
        st.markdown(f"- **Early Stopped:** `{run_info.get('early_stopped', False)}`")
        st.markdown(f"- **시작:** `{run_info['start_time']}`")
        st.markdown(f"- **종료:** `{run_info['end_time']}`")

    st.subheader("📊 최종 성능")
    st.metric("Train Accuracy", f"{run_info['train_accuracy']:.4f}")
    st.metric("Val Accuracy", f"{run_info['val_accuracy']:.4f}")
    st.metric("Test Accuracy", f"{run_info['test_accuracy']:.4f}")
    st.metric("Best Val Loss", f"{run_info['best_val_loss']:.4f}")
    st.markdown(f"**총 학습 시간:** `{run_info['duration_minutes']:.2f}분`")

    # --- 에폭별 시각화
    # epoch_logs = list(epochs.find({"run_id": selected_run}).sort("epoch", 1))
    # df = pd.DataFrame(epoch_logs)

    # st.subheader("📈 Epoch별 추이")
    # fig1, ax1 = plt.subplots()
    # ax1.plot(df["epoch"], df["train_acc"], label="Train Acc", color="skyblue")
    # ax1.plot(df["epoch"], df["val_acc"], label="Val Acc", color="orange")
    # ax1.set_title("Accuracy")
    # ax1.set_xlabel("Epoch")
    # ax1.set_ylabel("Accuracy")
    # ax1.legend()
    # st.pyplot(fig1)

    # fig2, ax2 = plt.subplots()
    # ax2.plot(df["epoch"], df["train_loss"], label="Train Loss", color="skyblue")
    # ax2.plot(df["epoch"], df["val_loss"], label="Val Loss", color="orange")
    # ax2.set_title("Loss")
    # ax2.set_xlabel("Epoch")
    # ax2.set_ylabel("Loss")
    # ax2.legend()
    # st.pyplot(fig2)

    # if "epoch_time_sec" in df.columns:
    #     fig3, ax3 = plt.subplots()
    #     ax3.plot(df["epoch"], df["epoch_time_sec"], color="green")
    #     ax3.set_title("Epoch Time per Epoch")
    #     ax3.set_xlabel("Epoch")
    #     ax3.set_ylabel("Seconds")
    #     st.pyplot(fig3)

    # # --- 원본 에폭 데이터
    # with st.expander("📄 원본 에폭 데이터 보기"):
    #     st.dataframe(df)
elif menu == "학습 상태 실시간 모니터링":
    st.title("📡 모델 학습 상태 실시간 모니터링")

    status_collection = db["status"]
    status_docs = list(status_collection.find())

    if not status_docs:
        st.info("📭 현재 상태 정보가 없습니다.")
        st.stop()

    for doc in status_docs:
        st.subheader(f"🧪 Run ID: `{doc['run_id']}`")
        st.markdown(f"- **Status:** `{doc['status']}`")
        st.markdown(f"- **Start Time:** `{doc.get('start_time', 'N/A')}`")
        if doc["status"] == "completed":
            st.markdown(f"- **End Time:** `{doc.get('end_time', 'N/A')}`")
            st.success("✅ 학습 완료!")
        elif doc["status"] == "in_progress":
            st.info("🌀 학습 진행 중...")

            # 진행률 표시
            # 현재 epoch 수 가져오기
            current_epoch = epochs.count_documents({"run_id": doc["run_id"]})
            # total_epoch = trainings.find_one({"run_id": doc["run_id"]})["epochs_run"] if trainings.find_one({"run_id": doc["run_id"]}) else 100
            total_epoch = 100

            st.markdown(f"**진행률:** {current_epoch} / {total_epoch} epochs")

            progress = min(int(current_epoch / total_epoch * 100), 100)
            st.progress(progress)

            # 실시간 갱신 유도
            st.caption("⏳ 새로고침하면 최신 상태가 반영됩니다.")
            st.button("🔄 새로고침", on_click=st.rerun)

        elif doc["status"] == "failed":
            st.error("❌ 학습 실패")

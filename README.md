# 🌾 Rice Leaf Disease Classifier & Monitoring Dashboard

> Transfer Learning 기반 쌀 잎 질병 분류 모델 성능 비교 및 실시간 모니터링 시스템 구축 프로젝트
> 
> 
> ✅ [실시간 모델 추적 대시보드 바로가기](https://model-tracker-bsmknozj45d9xndtbbeb4h.streamlit.app/#bd9568c6)
> 

---

## 📌 프로젝트 개요

본 프로젝트는 **쌀 잎에 발생하는 질병(Bacterial Blight, Brown Spot, Leaf Smut)**을 분류하기 위해 다양한 전이 학습 기반 CNN 모델을 실험하고, 학습 전 과정을 추적 및 시각화하는 시스템을 구축한 연구입니다.

- 사용 모델: `Custom CNN`, `ResNet50`, `VGG16`, `MobileNetV2`
- 성능 추적: `MongoDB + Streamlit 대시보드`
- 기술 스택: `PyTorch`, `MongoDB`, `Streamlit`, `Google Colab`

---

## 🔍 핵심 기능

### 1. 다양한 CNN 모델 비교 실험

- 사전학습된 모델 기반 Feature Extraction 및 Fine-tuning 적용
- 공통 구조: `GAP → Linear → BN → ReLU → Dropout → Linear`
- **최대 정확도**
    - ✅ `MobileNetV2`: 100%
    - ✅ `Custom CNN`: 100%
    - ✅ `VGG16`: 98.93%
    - ✅ `ResNet50`: 98.51%

### 2. 앙상블 학습 적용

- **Soft Voting** 방식으로 세 모델 결과 평균화
- `ResNet50 + VGG16 + MobileNetV2`
- 🎯 **최종 앙상블 정확도: 100% (Confusion Matrix 전 항목 완전 일치)**

### 3. 실시간 학습 모니터링 대시보드

- 학습 로그를 `MongoDB`에 저장하고 `Streamlit`으로 시각화
- 메뉴 구성:
    1. 모델 상세 보기
    2. 모델 비교 보기
    3. 시간 필터
    4. 학습 상태 실시간 확인

### 4. 실험 관리 자동화

- `insert_one + update_one` 방식으로 실시간 DB 갱신
- `streamlit.experimental_rerun` 문제 해결로 안정적 반영 구조 구축
- 향후 계획:
    - 🔧 `Grid Search` 기반 하이퍼파라미터 최적화
    - 📊 `Model Comparison Platform (MCP)` 기능 고도화

---

## 🗃 사용 데이터셋

- **📂 Kaggle: Rice Plant Diseases Dataset**
    
    [🔗 바로가기](https://www.kaggle.com/datasets/jay7080dev/rice-plant-diseases-dataset)
    
    - 총 4,684장 RGB 이미지 (224x224 리사이즈)
    - 클래스: `Bacterial Blight`, `Brown Spot`, `Leaf Smut`

---

## 🧪 실험 환경

| 항목 | 내용 |
| --- | --- |
| 프레임워크 | PyTorch |
| 학습 환경 | Google Colab Pro+ |
| 이미지 전처리 | Resize, Flip, Rotation, Jitter 등 |
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| Loss Function | CrossEntropyLoss |
| Early Stopping | patience=10, max_epoch=100 |
| DB | MongoDB (Atlas) |
| 시각화 | Streamlit Dashboard |
| 배포 | [🔗 Streamlit Cloud 링크](https://model-tracker-bsmknozj45d9xndtbbeb4h.streamlit.app/#bd9568c6) |
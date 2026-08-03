# 🫁 Evaluation of CNN Architectures for Pneumonia Classification in Chest X-Ray Images

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![DirectML](https://img.shields.io/badge/DirectML-GPU_Accelerated-0078D4?style=flat)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **PUCPR (Pontifícia Universidade Católica do Paraná)** 학부 연구/프로젝트  
> 흉부 X-ray 영상을 활용한 폐렴 이진 분류(NORMAL × PNEUMONIA)에서 **CNN 아키텍처**, **데이터 증강(Augmentation)**, **명암비 균일화(Equalization)**의 효과를 통계적으로 비교·분석한 연구 프로젝트입니다.

---

## 📌 Motivation
폐렴은 신속하고 정확한 진단이 필수적인 공중보건 과제입니다. 본 연구에서는 딥러닝 기반의 진단 보조 시스템 가능성을 탐색하기 위해 대표적인 CNN 아키텍처 3종(**ResNet50V2**, **DenseNet121**, **EfficientNetB0**)을 바탕으로 다음 두 가지 요인을 독립적으로 평가했습니다.

1. **Context A:** 전체 데이터셋 대상 **Data Augmentation**의 성능 영향 분석
2. **Context B:** 20% Stratified Sample 대상 **Contrast Equalization (Histogram & CLAHE)**의 성능 영향 분석

---

## 📊 Datasets
본 프로젝트는 **NORMAL**과 **PNEUMONIA** 클래스만을 활용합니다.

| Dataset | Source | Samples | Link |
| :--- | :--- | :---: | :---: |
| **Chest X-Ray Images** | Kaggle | 5,856 | [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| **ChestX-ray8** | NIH | 62,353 | [arXiv:1705.02315](https://arxiv.org/abs/1705.02315) |
| **COVID-19 Image Data** | IEEE | 764 (Filtered) | [GitHub Repository](https://github.com/ieee8023/covid-chestxray-dataset) |

* **Class Balancing:** 데이터 편향을 방지하기 위해 ChestX-ray8의 `NORMAL` 클래스는 최대 5,000장으로 캡핑(`chestxray8_normal_cap`) 처리했으며, 잔여 불균형은 학습 시 `class_weight`로 보정했습니다.
* **Directory Structure:** 데이터셋은 보안 및 용량 문제로 Git에 포함되지 않으며, `datasets/` 하위 디렉터리에 위치해야 합니다.

---

## ⚙️ Environment & Setup

- **OS:** Windows + Python 3.10
- **Acceleration:** DirectML (Tested on NVIDIA RTX 4050)
- **Framework:** `tensorflow-cpu==2.10.0` + `tensorflow-directml`

```bash
# Clone Repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Install Dependencies
pip install -r requirements.txt

# Verify DirectML GPU Adapter
python src/test_env.py

---

## 📁 Repository Structure
Plaintext
├── outputs/                  # 런타임 결과물 (EDA, Grad-CAM, Metrics, Weights)
│   ├── graphs/               # 분포, 증강 예시, Grad-CAM, 통계 검정 시각화
│   ├── logs/                 # Keras 및 CV Metrics (CSV/JSON)
│   └── models/               # Checkpoint 모델 가중치 (.ckpt)
├── src/
│   ├── main.py               # 전체 파이프라인 오케스트레이션 (Context A/B)
│   ├── preprocessing.py      # DataPipeline (Load, Clean, Balance, Split)
│   ├── training.py           # Stratified K-Fold CV & 2-Phase Fine-Tuning
│   ├── analysis.py           # Grad-CAM 및 Statistical Hypothesis Testing
│   └── test_env.py           # DirectML GPU 선택 진단 스크립트
└── requirements.txt

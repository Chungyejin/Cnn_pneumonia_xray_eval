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
```
---

## 📁 Repository Structure
```
.
├── outputs/                  # Experiment outputs (created at runtime)
│   ├── graphs/               # EDA distributions, augmentation/preprocessing examples,
│   │                         # contrast & intensity diagnostics, Grad-CAM, statistical plots
│   ├── logs/                 # Per-experiment cv_metrics.csv / test_metrics.csv|json + Keras logs
│   └── models/               # Trained checkpoints (TF weights-only format)
├── src/
│   ├── main.py               # Pipeline orchestration: Context A/B, metric re-extraction, Grad-CAM
│   ├── preprocessing.py      # DataPipeline: load, EDA, clean, balance, split, datagens
│   ├── training.py           # K-Fold CV + two-phase fine-tuning; fold/test evaluation
│   ├── analysis.py           # Diagnostics, Grad-CAM, statistical tests
│   └── test_env.py           # DirectML GPU-selection diagnostic
└── requirements.txt          # Project dependencies
```ies
```
## 🔬 Methodology (연구 방법론)
### 1. Data Splitting (데이터 분할)
전체 데이터의 15%를 Stratified 방식으로 사전 분리하여 고정 홀드아웃 테스트셋(Holdout Test Set)으로 보존합니다.

남은 85%의 데이터를 대상으로 StratifiedKFold (K=5)를 적용합니다.

Fold당 구성 비율: Train ~68% | Validation ~17% | Test 15%

### 2. Training (모델 학습)
ImageNet 사전 학습 가중치 기반의 전이 학습(Transfer Learning)을 적용하며, Fold당 2단계(Phase)로 학습합니다.

Phase 1: 백본(Backbone) 동결 및 분류기 헤드(Classifier Head)만 학습 (LR = 1e-3)

Phase 2: 백본의 마지막 30개 레이어(UNFREEZE_LAST = 30) 동결 해제 후 미세 조정 (LR = 1e-5). 단, ImageNet 통계 유지를 위해 BatchNormalization 레이어는 동결 유지.

val_loss 기반의 EarlyStopping(최적 가중치 복원) 및 ReduceLROnPlateau 적용

Fold별로 계산된 class_weight를 적용하여 클래스 불균형 보정

각 아키텍처별 전용 Preprocessing 레이어로 입력 데이터를 정규화하며, Data Generator는 [0, 255] 범위의 원본 이미지를 전달합니다.

데이터 증강 및 명암 대비 균일화는 학습 데이터에만 적용되며, 검증 및 테스트 데이터에는 항상 원본 이미지가 사용됩니다.

### 3. Experiments (실험 구성)
| Context | ContextExperiment (코드명) | Augmentation | Equalization |Equalization |
| :--- | :--- | :---: | :---: |:---:|
| Context A | baseline | No |No |Full Dataset|
| Context A | augmented | Yes |No |Full Dataset|
| Context B | baseline_sample | No |No |Stratified Sample|
| Context B | hist_sample | No |Global Histogram |20 % Stratified Sample|
| Context B | adaptive_sample | No |Adaptive (CLAHE) |20 % Stratified Sample|

- Context A: 전체 데이터셋 대상 데이터 증강의 독립적 효과 검증

- Context B: 20% 계층적 샘플 대상 명암 대비 균일화 기법의 독립적 효과 검증 (증강 미적용)

- 적용된 증강 기법: 회전(±15°), 가로/세로 이동(±5%), 확대/축소(±10%), 좌우 반전, 밝기 조절([0.85, 1.15]), fill_mode='nearest'

- 적용된 균일화 기법:

- Histogram: 전역 히스토그램 균일화 (cv2.equalizeHist)

- CLAHE: 적응형 히스토그램 균일화 (clipLimit=2.0, tileGridSize=(8, 8))

### 4. Analysis (결과 분석)
Diagnostics: 명암 대비 균일화 적용 전후의 클래스별 대비(픽셀 표준편차) 및 평균 명암값 분포 분석

Grad-CAM: 원본 vs 균일화 이미지에 대한 모델 어텐션 히트맵 비교

Statistical Tests:

Context A: 모델 아키텍처 비교 (Friedman 검정 및 Nemenyi 후속 검정)

Context B: 전처리/균일화 효과 비교 (Friedman 검정 및 Nemenyi 후속 검정)

Context A: 데이터 증강 효과 비교 (Wilcoxon 부호순위 검정)

📈 Results (실험 결과)
아래 수치는 5-Fold 교차 검증의 검증 폴드에서 측정된 f1_macro 메트릭의 평균 ± 표준편차(mean ± std)입니다. (Holdout Test-set 결과가 아님. 모든 셀에서 Accuracy 수치가 f1_macro와 ±0.002 이내로 매우 유사하여 f1_macro만 표기함)

#### Context A — Full Dataset (Augmentation Effect)
|Architecture|Baseline|Augmented|
| :--- | :--- | :---: |
|ResNet50V2|0.864 ± 0.004|0.875 ± 0.005|
|DenseNet121|0.868 ± 0.003|0.862 ± 0.003|
|EfficientNetB0|0.864 ± 0.007|0.870 ± 0.006|

데이터 증강 적용 시 ResNet50V2와 EfficientNetB0는 소폭의 성능 향상을 보였으나, DenseNet121에서는 약간의 성능 감소가 나타났습니다.

#### Context B — 20% Stratified Sample (Equalization Effect)
|Architecture|Baseline|Histogram Eq.|CLAHE
| :--- | :--- | :---: |:---: |
|ResNet50V2|0.851 ± 0.021|0.844 ± 0.018|0.829 ± 0.023|
|DenseNet121|0.852 ± 0.019|0.847 ± 0.019|0.836 ± 0.023|
|EfficientNetB0|0.845 ± 0.011|0.838 ± 0.028|0.833 ± 0.020|


Note on Wilcoxon Tests: > 5개의 검증 폴드(N=5)로 수행되는 2선택 대응 표본 Wilcoxon 부호순위 검정 구조상 이론적으로 산출 가능한 최소 p-value는 0.0625입니다. 따라서 효과 크기와 관계없이 유의수준 0.05 기준을 구조적으로 넘을 수 없어 "유의하지 않음" 결과가 나왔으며, 이는 데이터 증강 효과의 부재를 의미하기보다 제한된 통계적 검정력(Statistical Power)에 기인한 것입니다.

👥 Credits (팀원 및 소속)
Authors: Ana Flávia Martins Dos Santos, Isabella Vanderlinde Berkembrock, Michele Cristina Otta, Yejin Chung

Affiliation: PUCPR — Pontifícia Universidade Católica do Paraná (Curitiba, Brazil)

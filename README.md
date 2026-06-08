# Evaluation of CNN Architectures for Pneumonia Classification in Chest X-Ray Images

Applied deep learning project for binary classification of **pneumonia** in chest X-ray images, comparing ResNet, DenseNet, and EfficientNet architectures.

---

## Motivation

Pneumonia remains a major public health concern, demanding rapid and precise diagnosis for effective treatment and to prevent complications. Deep learning-based diagnostic support systems have shown strong potential for accelerating and standardizing clinical screening.

This project evaluates three well-established CNN architectures — **ResNet50V2**, **DenseNet121**, and **EfficientNetB0** — for binary classification (NORMAL × PNEUMONIA), studying two factors separately: the effect of **data augmentation** (on the full dataset) and the effect of **histogram-based contrast equalization** (on a stratified 20% sample).

---

## Dataset

Only the NORMAL and PNEUMONIA categories are considered.

1. **Chest X-Ray Images (Pneumonia)**
   - **Source:** *Chest X-Ray Images (Pneumonia)* (Kaggle)
   - **Size:** 5,863 images
   - **Access:** <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>

2. **ChestX-ray8**
   - **Source:** *ChestX-ray8* — National Institute of Health (NIH)
   - **Size:** 62,353 images
   - **Access:** <https://arxiv.org/abs/1705.02315>

3. **COVID-19 Image Data Collection**
   - **Source:** *COVID-19 image data collection* (IEEE)
   - **Size:** 764 images
   - **Access:** <https://github.com/ieee8023/covid-chestxray-dataset>

> **Balancing note:** the NORMAL class from ChestX-ray8 is capped at 5,000 samples (`chestxray8_normal_cap`) to avoid skewing the unified dataset. Residual class imbalance is handled via `class_weight` during training.

> The datasets are not committed to the repository. The pipeline expects them under a `datasets/` directory, with class inferred from folder names containing `NORMAL` or `PNEUMONIA`.

---

## Environment

The project was developed for **Windows + Python 3.10**, with GPU acceleration through **DirectML** (tested on an RTX 4050).

- `tensorflow-cpu==2.10.0` provides the base; the GPU is enabled via the DirectML plugin.
- Adapter selection is done with the `DML_VISIBLE_DEVICES` environment variable, set **before** importing TensorFlow (see `main.py` / `test_env.py`).
- Run `test_env.py` standalone to confirm which adapter the chosen index selects before training.

Install dependencies with:

```bash
pip install -r requirements.txt
```

Full dependency list is in `requirements.txt` (OpenCV, NumPy, pandas, scikit-learn, matplotlib, SciPy, scikit-posthocs).

---

## Repository structure

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
```

> Note: the exact `src/` layout above reflects the source modules in this project; adjust the path prefix if your local tree differs.

---

## Methodology

**Data splitting**
- 15% reserved as a fixed stratified holdout test set (separated before any fold).
- Remaining 85% split via `StratifiedKFold` with K=5.
- Per fold: ~68% train | ~17% validation | 15% test.

**Training**
- Transfer learning with ImageNet weights, in **two phases per fold**:
  - *Phase 1:* train only the classifier head (backbone frozen), LR = 1e-3.
  - *Phase 2:* fine-tune the last `UNFREEZE_LAST` (30) backbone layers, LR = 1e-5; `BatchNormalization` stays frozen to preserve ImageNet statistics.
- `EarlyStopping` (restore best weights) and `ReduceLROnPlateau` on `val_loss`.
- Class imbalance handled per fold via balanced `class_weight`.
- Each architecture normalizes inputs through its own registered preprocessing layer; data generators deliver raw `[0, 255]` images (no rescale).

**Experiments (as implemented)**

The pipeline runs in two contexts:

| Context | Experiment (code name) | Augmentation | Equalization | Dataset |
|---|---|---|---|---|
| A | `baseline`         | No  | No                | Full |
| A | `augmented`        | Yes | No                | Full |
| B | `baseline_sample`  | No  | No                | 20% stratified sample |
| B | `hist_sample`      | No  | Global histogram  | 20% stratified sample |
| B | `adaptive_sample`  | No  | Adaptive (CLAHE)  | 20% stratified sample |

> Context A isolates the **augmentation** effect on the full dataset; Context B isolates the **equalization** effect on a 20% stratified sample. Equalization and augmentation are evaluated separately (the equalization experiments do not apply augmentation).

**Augmentation applied** (when active): ±15° rotation, ±5% horizontal/vertical shift, ±10% zoom, horizontal flip, brightness in `[0.85, 1.15]` (≈ ±15%), `fill_mode='nearest'`.

**Equalization**
- *Histogram:* global histogram equalization (`cv2.equalizeHist`).
- *CLAHE:* adaptive equalization (`clipLimit=2.0`, `tileGridSize=(8, 8)`).

---

## Analysis

- **Diagnostics:** per-class contrast (pixel std) and mean intensity distributions, before and after each equalization method.
- **Grad-CAM:** model-attention heatmaps comparing original vs. equalized inputs.
- **Statistical tests:**
  - Architecture comparison (Friedman, with Nemenyi post-hoc) — Context A.
  - Preprocessing/equalization effect (Friedman, with Nemenyi post-hoc) — Context B.
  - Augmentation effect (Wilcoxon signed-rank) — Context A.

---

## Credits

Authors: Ana Flávia Martins Dos Santos; Isabella Vanderlinde Berkembrock; Michele Cristina Otta; Yejin Chung
Affiliation: PUCPR — Pontifícia Universidade Católica do Paraná (Curitiba, Brazil)

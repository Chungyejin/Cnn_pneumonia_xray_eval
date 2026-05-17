import os
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocessing import DataPipeline
from training import (
    train_fold, evaluate_fold, evaluate_test_set,
    SEED, TARGET_SIZE, ARCHITECTURES, MODELS_DIR, LOGS_DIR
)

print("TensorFlow:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Preprocessing pipeline (splits, folds, datagens)

pipeline = DataPipeline(
    base_path='datasets', target_size=TARGET_SIZE, graphs_dir='outputs/graphs',
    random_state=SEED, chestxray8_normal_cap=5000, n_splits=5, test_size=0.15
)
pipeline.run()

folds          = pipeline.folds
X_test, y_test = pipeline.X_test, pipeline.y_test
val_datagen    = pipeline.val_test_datagen   # pass-through — shared by all experiments

# Baseline: no augmentation, pass-through (architecture-specific preprocessing
# is applied inside the model — see training.py::ArchPreprocessing).
baseline_datagen = ImageDataGenerator()

# Preprocessed: with full augmentation (config in preprocessing.py)
augmented_datagen = pipeline.train_datagen

# Augmentation-light: hypothesis test for DenseNet121.
#   In the first comparison, augmentation HURT DenseNet121 (−0.75 pp F1 macro
#   across all 5 folds) while HELPING ResNet50V2 (+0.82 pp). One candidate
#   explanation is that brightness ±15% corrupts low-level features that
#   DenseNet's dense connectivity propagates throughout the network. Here we
#   drop brightness, halve the geometric transforms, and re-test — if F1 macro
#   returns close to baseline, the hypothesis is supported.
aug_light_datagen = ImageDataGenerator(
    rotation_range     = 8,
    width_shift_range  = 0.025,
    height_shift_range = 0.025,
    zoom_range         = 0.05,
    horizontal_flip    = True,
    fill_mode          = 'nearest'
)

# Experiment runner

def run_experiment(experiment: str, train_datagen, archs=ARCHITECTURES) -> tuple:
    """
    Runs K-Fold training + test evaluation for the given architectures.

    Parameters
    ----------
    experiment    : label used in checkpoints and log filenames
                    ('baseline', 'preprocessed', 'aug_light', ...)
    train_datagen : ImageDataGenerator used for training folds
    archs         : iterable of architecture names to run (defaults to all)

    Returns
    -------
    cv_metrics, test_metrics : lists of metric dicts
    """
    exp_models_dir = f'{MODELS_DIR}/{experiment}'
    exp_logs_dir   = f'{LOGS_DIR}/{experiment}'
    os.makedirs(exp_models_dir, exist_ok=True)
    os.makedirs(exp_logs_dir,   exist_ok=True)

    cv_metrics, test_metrics = [], []

    for arch in archs:
        print(f"\n{'#'*55}\n  [{experiment.upper()}] {arch.upper()}\n{'#'*55}")
        arch_cv = []

        for fold_data in folds:
            result  = train_fold(arch, fold_data, train_datagen, val_datagen,
                                 experiment=experiment)
            metrics = evaluate_fold(result)
            arch_cv.append(metrics)
            cv_metrics.append(metrics)

        # Best fold -> test set evaluation
        best      = max(arch_cv, key=lambda m: m['f1_macro'])
        ckpt_path = f"{exp_models_dir}/{arch}_fold{best['fold']}_phase2_best.keras"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Best-fold checkpoint missing: {ckpt_path}. "
                f"Refusing to evaluate the wrong model on the test set."
            )
        model = tf.keras.models.load_model(ckpt_path)

        t = evaluate_test_set(model, X_test, y_test, val_datagen, arch)
        test_metrics.append(t)

    # Save per-experiment results
    pd.DataFrame(cv_metrics).to_csv(f'{exp_logs_dir}/cv_metrics.csv',    index=False)
    pd.DataFrame(test_metrics).to_csv(f'{exp_logs_dir}/test_metrics.csv', index=False)
    json.dump(test_metrics, open(f'{exp_logs_dir}/test_metrics.json', 'w'), indent=2)
    print(f"\n[{experiment}] Results saved to '{exp_logs_dir}/'")

    return cv_metrics, test_metrics

# Run all experiments

print("\n" + "="*55)
print("  EXPERIMENT 1: BASELINE (no augmentation)")
print("="*55)
baseline_cv, baseline_test = run_experiment('baseline', baseline_datagen)

print("\n" + "="*55)
print("  EXPERIMENT 2: PREPROCESSED (with augmentation)")
print("="*55)
preprocessed_cv, preprocessed_test = run_experiment('preprocessed', augmented_datagen)

print("\n" + "="*55)
print("  EXPERIMENT 3: AUG_LIGHT (DenseNet hypothesis test)")
print("="*55)
# Restricted to DenseNet121: this experiment exists specifically to test whether
# lighter augmentation recovers the performance loss observed under full
# augmentation. Running the other architectures here would cost compute without
# answering the research question.
aug_light_cv, aug_light_test = run_experiment(
    'aug_light', aug_light_datagen, archs=['densenet121']
)
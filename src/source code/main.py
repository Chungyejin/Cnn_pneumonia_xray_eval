# main.py — orquestrador do pipeline.
# Ordem sugerida: Context A -> Context B -> reextract -> testes -> Grad-CAM.

import os

# Seleção de GPU (DirectML) DEVE vir antes de `import tensorflow`.
# DML_VISIBLE_DEVICES usa índices de adaptador DXGI/DirectML (NÃO os números do
# Gerenciador de Tarefas). Nesta máquina: adaptador 0 = RTX 4050. Só o adaptador
# escolhido fica visível e é reindexado como GPU:0. Reveja com gpu_probe.py se o HW mudar.
os.environ.setdefault("DML_VISIBLE_DEVICES", "0")

import json

import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

print("TensorFlow:", tf.__version__)
print("DML_VISIBLE_DEVICES:", os.environ.get("DML_VISIBLE_DEVICES"))
print("Visible GPU devices:", tf.config.list_physical_devices('GPU'))

from analysis import (analyze_contrast, analyze_intensity_distribution,
                      reextract_cv_metrics, reextract_test_metrics,
                      run_gradcam, run_statistical_tests,
                      verify_datagen_preprocessing)
from preprocessing import DataPipeline
from training import (ARCHITECTURES, LOGS_DIR, MODELS_DIR, SEED, TARGET_SIZE,
                      evaluate_fold, evaluate_test_set, load_trained_model,
                      train_fold)

# Flags de controle — False pula a seção.
RUN_CONTEXT_A          = False   # experimentos no dataset completo
RUN_CONTEXT_B          = False  # experimentos na amostra estratificada de 20%
REEXTRACT_CV_METRICS   = False  # reconstrói cv_metrics.csv dos checkpoints
REEXTRACT_TEST_METRICS = False  # reconstrói test_metrics.csv dos checkpoints
RUN_STATISTICAL_TESTS  = True  # requer CSVs do context A e/ou B
RUN_DIAGNOSTICS        = True  # contraste + intensidade (sem GPU)
VERIFY_DATAGEN         = True  # confirma preprocessing_function ativa
RUN_GRADCAM            = True  # requer modelo treinado; ative após A/B

SAMPLE_FRAC = 0.20  # fração usada no Context B

# DEBUG_SAMPLE=True: roda Context A em ~2% do dataset, só EfficientNetB0 e fold 1
# (arquitetura que quebrava a serialização — testa o pior caso) e valida save/load
# do checkpoint + uso da GPU. Defina False para o treino real.
DEBUG_SAMPLE = False
DEBUG_FRAC   = 0.02

pipeline = DataPipeline(
    base_path             = 'datasets',
    target_size           = TARGET_SIZE,
    graphs_dir            = 'outputs/graphs',
    random_state          = SEED,
    chestxray8_normal_cap = 5000,
    n_splits              = 5,
    test_size             = 0.15,
    skip_plots            = DEBUG_SAMPLE,   # pula plots de EDA no debug
)
pipeline.run()

folds          = pipeline.folds
X_test, y_test = pipeline.X_test, pipeline.y_test

# Datagen baseline — usado na validação e no teste em todos os experimentos.
val_datagen = pipeline.get_datagen_for_experiment('baseline')

if VERIFY_DATAGEN:
    sample_df = (pipeline.df[pipeline.df['height'].notna()]
                 .groupby('label', group_keys=False)
                 .apply(lambda g: g.sample(min(50, len(g)), random_state=SEED)))
    verify_datagen_preprocessing(pipeline, sample_df[['path', 'label']],
                                 target_size=TARGET_SIZE,
                                 graphs_dir='outputs/graphs')

if RUN_DIAGNOSTICS:
    analyze_contrast(pipeline.df, graphs_dir='outputs/graphs')
    analyze_intensity_distribution(pipeline.df, graphs_dir='outputs/graphs')


def sample_folds(folds, X_test, y_test, frac, random_state=SEED):
    # Subamostra estratificada frac de cada fold e do teste; compartilhada no Context B.
    sampled_folds = []
    for fold_data in folds:
        _, X_tr, _, y_tr = train_test_split(
            fold_data['X_train'], fold_data['y_train'],
            test_size=frac, stratify=fold_data['y_train'], random_state=random_state,
        )
        _, X_vl, _, y_vl = train_test_split(
            fold_data['X_val'], fold_data['y_val'],
            test_size=frac, stratify=fold_data['y_val'], random_state=random_state,
        )
        sampled_folds.append({
            'fold'   : fold_data['fold'],
            'X_train': X_tr, 'y_train': y_tr,
            'X_val'  : X_vl, 'y_val'  : y_vl,
        })

    _, X_ts, _, y_ts = train_test_split(
        X_test, y_test,
        test_size=frac, stratify=y_test, random_state=random_state,
    )

    n_train = sum(len(f['X_train']) for f in sampled_folds) // len(sampled_folds)
    n_val   = sum(len(f['X_val'])   for f in sampled_folds) // len(sampled_folds)
    print(f"\n[sample_folds] frac={frac} | "
          f"~{n_train} train/fold | ~{n_val} val/fold | {len(X_ts)} test")

    return sampled_folds, X_ts, y_ts


def run_experiment(name, train_datagen, folds, X_test, y_test,
                   archs_override=None, folds_override=None):
    # K-Fold + avaliação no teste para cada arquitetura. Seleciona o melhor fold por
    # f1_macro e avalia no teste. Salva cv/test metrics e checkpoints.
    models_dir = f'{MODELS_DIR}/{name}'
    logs_dir   = f'{LOGS_DIR}/{name}'
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir,   exist_ok=True)

    active_archs = archs_override if archs_override is not None else ARCHITECTURES
    active_folds = folds_override if folds_override is not None else folds

    cv_metrics, test_metrics = [], []

    for arch in active_archs:
        print(f"\n{'#'*55}\n  [{name.upper()}] {arch.upper()}\n{'#'*55}")
        arch_cv = []

        # Mantém o melhor modelo em memória. train_fold usa EarlyStopping com
        # restore_best_weights=True, então result['model'] já tem os melhores pesos
        # do fold (igual ao checkpoint). Usar direto evita reload de disco, frágil
        # para o backbone aninhado no loader HDF5 do TF 2.10. Checkpoints seguem salvos.
        best_model = None
        best_f1    = -1.0
        best_fold  = None

        for fold_data in active_folds:
            result  = train_fold(arch, fold_data, train_datagen, val_datagen,
                                 experiment=name)
            metrics = evaluate_fold(result)
            arch_cv.append(metrics)
            cv_metrics.append(metrics)

            if metrics['f1_macro'] > best_f1:
                best_f1    = metrics['f1_macro']
                best_model = result['model']   # guarda só o melhor; resto é coletado
                best_fold  = metrics['fold']

        if best_model is None:
            raise RuntimeError(f"No fold trained for '{arch}' — cannot evaluate test set.")

        print(f"\n  [{name}] {arch} best fold: {best_fold} | f1_macro={best_f1:.4f}")
        t = evaluate_test_set(best_model, X_test, y_test, val_datagen, arch)
        test_metrics.append(t)

    pd.DataFrame(cv_metrics).to_csv(f'{logs_dir}/cv_metrics.csv',    index=False)
    pd.DataFrame(test_metrics).to_csv(f'{logs_dir}/test_metrics.csv', index=False)
    with open(f'{logs_dir}/test_metrics.json', 'w') as _f:
        json.dump(test_metrics, _f, indent=2)
    print(f"\n[{name}] Results saved to '{logs_dir}/'")

    return cv_metrics, test_metrics


# Context A — dataset completo
if RUN_CONTEXT_A:
    print("\n" + "="*55)
    print("  CONTEXT A — full dataset")
    print("="*55)

    if DEBUG_SAMPLE:
        # Debug: amostra mínima + só EfficientNetB0 + só fold 1 (testa o caminho de
        # falha de serialização e confirma save/load + uso de GPU em minutos).
        print(f"\n  [DEBUG] {int(DEBUG_FRAC * 100)}% sample | arch=efficientnetb0 | fold=1 only")
        _folds_a, _X_test_a, _y_test_a = sample_folds(
            folds, X_test, y_test, frac=DEBUG_FRAC
        )
        _debug_archs = ['efficientnetb0']
        _debug_folds = [_folds_a[0]]   # só fold 1

        run_experiment('baseline',
                       pipeline.get_datagen_for_experiment('baseline'),
                       _folds_a, _X_test_a, _y_test_a,
                       archs_override=_debug_archs,
                       folds_override=_debug_folds)

        # Valida o ciclo save+load em disco para o EfficientNetB0 (único passo que
        # confirma o reload do checkpoint formato TF; o treino normal usa o modelo
        # em memória e não recarrega).
        _ckpt = f'{MODELS_DIR}/baseline/efficientnetb0_fold1_phase2_best'
        print("\n  [DEBUG] Validating checkpoint reload...")
        if os.path.exists(_ckpt + '.index'):
            _m = load_trained_model('efficientnetb0', _ckpt)
            print(f"    OK: reloaded {_ckpt} ({len(_m.weights)} weight tensors)")
        else:
            print(f"    WARNING: checkpoint not found at '{_ckpt}.index' — "
                  "check that training saved it.")

        print("\n  [DEBUG] Done. Check:")
        print("    1. 'OK: reloaded ...' printed just above (save+load works)")
        print("    2. No serialization errors during training/test eval above")
        print("    3. RTX 4050 shown in Gerenciador de Tarefas during training")
        print("  Set DEBUG_SAMPLE = False to run full Context A.")

    else:
        _folds_a, _X_test_a, _y_test_a = folds, X_test, y_test

        run_experiment('baseline',
                       pipeline.get_datagen_for_experiment('baseline'),
                       _folds_a, _X_test_a, _y_test_a)

        run_experiment('augmented',
                       pipeline.get_datagen_for_experiment('augmented'),
                       _folds_a, _X_test_a, _y_test_a)

# Context B — amostra estratificada de 20% (diagnóstico de equalização)
if RUN_CONTEXT_B:
    print("\n" + "="*55)
    print(f"  CONTEXT B — {int(SAMPLE_FRAC * 100)}% stratified sample")
    print("="*55)

    folds_s, X_test_s, y_test_s = sample_folds(
        folds, X_test, y_test, frac=SAMPLE_FRAC
    )

    for exp_name, mode in [
        ('baseline_sample', 'baseline'),
        ('hist_sample',     'hist'),
        ('adaptive_sample', 'adaptive'),
    ]:
        print(f"\n  Experiment: {exp_name}")
        run_experiment(exp_name,
                       pipeline.get_datagen_for_experiment(mode),
                       folds_s, X_test_s, y_test_s)

# Reextração de métricas — só faz sentido após rodar o Context A ao menos uma vez.
if REEXTRACT_CV_METRICS:
    for exp in ['baseline', 'augmented']:
        reextract_cv_metrics(exp, pipeline, val_datagen,
                             logs_dir=LOGS_DIR, models_dir=MODELS_DIR)

if REEXTRACT_TEST_METRICS:
    for exp in ['baseline', 'augmented']:
        reextract_test_metrics(exp, pipeline, val_datagen,
                               logs_dir=LOGS_DIR, models_dir=MODELS_DIR)

# Testes estatísticos
if RUN_STATISTICAL_TESTS:
    run_statistical_tests(logs_dir=LOGS_DIR, graphs_dir='outputs/graphs')

# Grad-CAM — ative após o treino.
if RUN_GRADCAM:
    # Checkpoint weights-only no formato TF: prefixo SEM extensão (load_trained_model
    # chama load_weights). O .keras anterior quebrava o reload.
    best_model_path = 'outputs/models/baseline/resnet50v2_fold1_phase2_best'
    model           = load_trained_model('resnet50v2', best_model_path)

    sample_df = (pipeline.df[pipeline.df['height'].notna()]
                 .groupby('label', group_keys=False)
                 .apply(lambda g: g.sample(2, random_state=SEED)))

    run_gradcam(
        model       = model,
        image_paths = sample_df['path'].tolist(),
        labels      = sample_df['label'].tolist(),
        target_size = TARGET_SIZE,
        graphs_dir  = 'outputs/graphs',
        n_samples   = 4,
    )

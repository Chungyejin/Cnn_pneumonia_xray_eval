# training.py — K-Fold CV + fine-tuning em duas fases.
# Arquiteturas: ResNet50V2 | DenseNet121 | EfficientNetB0.
# Fase 1: só a cabeça (backbone congelado). Fase 2: últimas UNFREEZE_LAST camadas,
# LR menor; BatchNorm fica sempre congelado (preserva estatísticas da ImageNet).
# Datagens entregam [0,255] sem rescale; cada modelo normaliza na sua preprocess layer.

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import DenseNet121, EfficientNetB0, ResNet50V2
from tensorflow.keras.applications import densenet, efficientnet, resnet_v2
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping,
                                         ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                      GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                              precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight

SEED          = 42
TARGET_SIZE   = (224, 224)
BATCH_SIZE    = 32
MODELS_DIR    = 'outputs/models'
LOGS_DIR      = 'outputs/logs'

EPOCHS_P1     = 10   # fase só-cabeça
EPOCHS_P2     = 20   # fase de fine-tuning
UNFREEZE_LAST = 30   # nº de camadas do backbone descongeladas na fase 2
LR_P1         = 1e-3
LR_P2         = 1e-5

# Keras atribui índices alfabéticos: NORMAL→0, PNEUMONIA→1. LABEL_MAP espelha isso.
LABEL_MAP   = {'NORMAL': 0, 'PNEUMONIA': 1}
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

ARCHITECTURES = ['resnet50v2', 'densenet121', 'efficientnetb0']

tf.random.set_seed(SEED)
np.random.seed(SEED)

for _d in [MODELS_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)


# Camadas de preprocess registradas (subclasses no lugar de Lambda) para que
# load_model funcione no Keras 3 sem custom_objects.

@tf.keras.utils.register_keras_serializable(package='pneumonia')
class ResNetPreprocess(tf.keras.layers.Layer):
    def call(self, x):
        return resnet_v2.preprocess_input(x)


@tf.keras.utils.register_keras_serializable(package='pneumonia')
class DenseNetPreprocess(tf.keras.layers.Layer):
    def call(self, x):
        return densenet.preprocess_input(x)


@tf.keras.utils.register_keras_serializable(package='pneumonia')
class EfficientNetPreprocess(tf.keras.layers.Layer):
    def call(self, x):
        return efficientnet.preprocess_input(x)


# nome -> (classe do backbone, classe da preprocess layer)
_ARCH_REGISTRY = {
    'resnet50v2'    : (ResNet50V2,     ResNetPreprocess),
    'densenet121'   : (DenseNet121,    DenseNetPreprocess),
    'efficientnetb0': (EfficientNetB0, EfficientNetPreprocess),
}


def build_model(arch: str, freeze_base: bool = True) -> tuple:
    # Retorna (model completo, base) — a base é guardada p/ descongelar na fase 2.
    if arch not in _ARCH_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from: {list(_ARCH_REGISTRY)}")

    backbone_cls, preprocess_cls = _ARCH_REGISTRY[arch]
    base           = backbone_cls(include_top=False, weights='imagenet',
                                  input_shape=(*TARGET_SIZE, 3))
    base.trainable = not freeze_base

    inputs = tf.keras.Input(shape=(*TARGET_SIZE, 3))
    x      = preprocess_cls(name='preprocess')(inputs)  # registrada, serializa ok
    x      = base(x, training=False)

    x   = GlobalAveragePooling2D()(x)
    x   = BatchNormalization()(x)
    x   = Dense(256, activation='relu')(x)
    x   = Dropout(0.4)(x)
    x   = Dense(128, activation='relu')(x)
    x   = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=out, name=arch)
    return model, base


def unfreeze_top_layers(base: tf.keras.Model, n: int = UNFREEZE_LAST) -> None:
    # Descongela as últimas n camadas do backbone; BN permanece congelado.
    base.trainable = True

    for layer in base.layers[: len(base.layers) - n]:
        layer.trainable = False

    for layer in base.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False


def load_trained_model(arch: str, weights_path: str) -> tf.keras.Model:
    # Reconstrói arch e carrega checkpoint weights-only no formato TF (sem extensão:
    # <weights_path>.index / .data-*). Não usar load_model: salvar o modelo completo
    # serializa o config p/ JSON e falha no EfficientNet (Normalization com tensores).
    # O formato TF restaura por grafo de objetos e lida com o backbone aninhado.
    model, _ = build_model(arch, freeze_base=True)
    # expect_partial(): o checkpoint carrega o estado do otimizador (Adam m/v, lr,
    # iter) que não existe neste modelo de inferência sem compile; silencia os
    # warnings de "unrestored values". Os pesos das camadas são restaurados normal.
    model.load_weights(weights_path).expect_partial()
    return model


def make_generator(datagen, X, y, shuffle: bool = True):
    # Ordem de classe: Keras indexa alfabeticamente -> NORMAL=0, PNEUMONIA=1.
    return datagen.flow_from_dataframe(
        dataframe   = pd.DataFrame({'filename': X, 'class': y}),
        x_col       = 'filename',
        y_col       = 'class',
        target_size = TARGET_SIZE,
        color_mode  = 'rgb',
        class_mode  = 'binary',
        batch_size  = BATCH_SIZE,
        shuffle     = shuffle,
        seed        = SEED,
    )


def get_class_weights(y_train):
    # Pesos balanceados para o desbalanceamento residual de classe.
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return {LABEL_MAP[c]: w for c, w in zip(classes, weights)}


class SanitizeLogs(tf.keras.callbacks.Callback):
    # Converte cada valor de `logs` para float antes de CSVLogger/ModelCheckpoint.
    # Sob TF 2.10 + DirectML alguns valores chegam como EagerTensor/array e quebram
    # a serialização JSON/CSV desses callbacks.
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        for key in list(logs.keys()):
            value = logs[key]
            if hasattr(value, 'numpy'):
                value = value.numpy()
            arr = np.asarray(value).reshape(-1)
            logs[key] = float(arr[0]) if arr.size else 0.0


def get_callbacks(arch: str, fold: int, phase: int, experiment: str = ''):
    exp_models = f'{MODELS_DIR}/{experiment}' if experiment else MODELS_DIR
    exp_logs   = f'{LOGS_DIR}/{experiment}'   if experiment else LOGS_DIR
    os.makedirs(exp_models, exist_ok=True)
    os.makedirs(exp_logs,   exist_ok=True)

    prefix = f'{exp_models}/{arch}_fold{fold}_phase{phase}'
    return [
        # SanitizeLogs primeiro: limpa EagerTensor/array de `logs` antes do resto.
        SanitizeLogs(),
        EarlyStopping(monitor='val_loss', patience=5 if phase == 1 else 8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3 if phase == 1 else 5, min_lr=1e-7, verbose=1),
        # save_weights_only=True + caminho SEM extensão -> checkpoint formato TF
        # (.index/.data-*), não HDF5. Evita falha de JSON (EfficientNet) e o erro
        # "axes don't match array" do loader HDF5 com backbone aninhado no TF 2.10.
        ModelCheckpoint(f'{prefix}_best', monitor='val_loss',
                        save_best_only=True, save_weights_only=True, verbose=0),
        CSVLogger(f'{exp_logs}/{arch}_fold{fold}_phase{phase}.csv'),
    ]


def train_fold(arch: str, fold_data: dict, train_datagen, val_datagen,
               experiment: str = '') -> dict:
    # Treina um fold em duas fases e devolve o necessário p/ avaliação.
    fold             = fold_data['fold']
    X_train, y_train = fold_data['X_train'], fold_data['y_train']
    X_val,   y_val   = fold_data['X_val'],   fold_data['y_val']

    print(f"\n{'='*55}\n  Fold {fold} | {arch.upper()}\n{'='*55}")

    cw        = get_class_weights(y_train)
    train_gen = make_generator(train_datagen, X_train, y_train, shuffle=True)
    val_gen   = make_generator(val_datagen,   X_val,   y_val,   shuffle=False)
    steps_tr  = max(1, len(X_train) // BATCH_SIZE)
    steps_val = max(1, len(X_val)   // BATCH_SIZE)

    print(f"  Class weights: {cw}")

    def compile_and_fit(model, lr, epochs, phase):
        model.compile(
            optimizer = Adam(lr),
            loss      = 'binary_crossentropy',
            metrics   = ['accuracy',
                          tf.keras.metrics.Precision(name='precision'),
                          tf.keras.metrics.Recall(name='recall')],
        )
        return model.fit(
            train_gen,
            steps_per_epoch  = steps_tr,
            validation_data  = val_gen,
            validation_steps = steps_val,
            epochs           = epochs,
            class_weight     = cw,
            callbacks        = get_callbacks(arch, fold, phase, experiment),
            verbose          = 1,
        )

    print(f"\n  [Phase 1] Head only — LR={LR_P1}")
    model, base = build_model(arch, freeze_base=True)
    hist1       = compile_and_fit(model, LR_P1, EPOCHS_P1, phase=1)

    print(f"\n  [Phase 2] Fine-tuning — LR={LR_P2}")
    unfreeze_top_layers(base)
    hist2 = compile_and_fit(model, LR_P2, EPOCHS_P2, phase=2)

    return dict(
        fold       = fold,
        arch       = arch,
        model      = model,
        history_p1 = hist1.history,
        history_p2 = hist2.history,
        val_gen    = val_gen,
        y_val      = y_val,
    )


def predict(model, gen, y_true, threshold=0.5):
    # gen com shuffle=False -> ordem casa com y_true. Fatia y_prob p/ descartar
    # padding do último batch (menor).
    n_steps    = int(np.ceil(len(y_true) / gen.batch_size))
    gen.reset()
    y_prob     = model.predict(gen, steps=n_steps, verbose=0).flatten()
    y_prob     = y_prob[:len(y_true)]
    y_pred     = (y_prob >= threshold).astype(int)
    y_true_int = np.array([LABEL_MAP[lbl] for lbl in y_true])
    return y_true_int, y_pred


def compute_metrics(y_true, y_pred, fold, arch) -> dict:
    return {
        'fold'              : fold,
        'arch'              : arch,
        'accuracy'          : accuracy_score (y_true, y_pred),
        'precision_macro'   : precision_score(y_true, y_pred, average='macro',    zero_division=0),
        'recall_macro'      : recall_score   (y_true, y_pred, average='macro',    zero_division=0),
        'f1_macro'          : f1_score       (y_true, y_pred, average='macro',    zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted'   : recall_score   (y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted'       : f1_score       (y_true, y_pred, average='weighted', zero_division=0),
    }


def evaluate_fold(result: dict) -> dict:
    y_true, y_pred = predict(result['model'], result['val_gen'], result['y_val'])
    return compute_metrics(y_true, y_pred, result['fold'], result['arch'])


def evaluate_test_set(model, X_test, y_test, val_datagen, arch: str) -> dict:
    test_gen       = make_generator(val_datagen, X_test, y_test, shuffle=False)
    y_true, y_pred = predict(model, test_gen, y_test)
    metrics        = compute_metrics(y_true, y_pred, fold=-1, arch=arch)
    metrics['split'] = 'test'
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    return metrics

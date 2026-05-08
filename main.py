import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from preprocessing.preprocessing import DataPipeline

print("TensorFlow:", tf.__version__)
print("GPU disponível:", tf.config.list_physical_devices('GPU'))

pipeline = DataPipeline(
    base_path             = 'datasets',
    target_size           = (224, 224),
    graphs_dir            = 'graphs',
    random_state          = 42,
    chestxray8_normal_cap = 5000,
    n_splits              = 5,
    test_size             = 0.15
)

pipeline.run()

# --- Resultados disponíveis para as próximas etapas ---

# DataFrame completo com metadados (pós-amostragem)
df = pipeline.df

# Holdout test set — usado apenas na avaliação final
X_test = pipeline.X_test
y_test = pipeline.y_test

# Geradores de imagem
train_datagen    = pipeline.train_datagen     # com augmentation
val_test_datagen = pipeline.val_test_datagen  # sem augmentation

# Folds para cross-validation
# Cada fold é um dict com chaves: 'fold', 'X_train', 'y_train', 'X_val', 'y_val'
folds = pipeline.folds

# Exemplo de acesso ao fold 1
fold_1   = folds[0]
X_train  = fold_1['X_train']
y_train  = fold_1['y_train']
X_val    = fold_1['X_val']
y_val    = fold_1['y_val']

# Class weights calculados sobre o treino do fold 1
# (recalcular a cada fold para refletir a distribuição exata daquele treino)
classes     = np.unique(y_train)
pesos       = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight = dict(zip(classes, pesos))
print("\nClass weights (fold 1):", class_weight)

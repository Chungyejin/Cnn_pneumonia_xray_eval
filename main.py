import tensorflow as tf
from preprocessing.preprocessing import DataPipeline

print("TensorFlow:", tf.__version__)
print("GPU disponível:", tf.config.list_physical_devices('GPU'))

pipeline = DataPipeline(
    base_path    = 'datasets',
    target_size  = (224, 224),  # padrão para VGG, ResNet, EfficientNet
    graphs_dir   = 'graphs',
    random_state = 42           # reprodutibilidade dos splits
)

pipeline.run()

# Resultados disponíveis para as próximas etapas
df               = pipeline.df           # DataFrame completo com metadados
X_train, y_train = pipeline.X_train, pipeline.y_train
X_val,   y_val   = pipeline.X_val,   pipeline.y_val
X_test,  y_test  = pipeline.X_test,  pipeline.y_test
train_datagen    = pipeline.train_datagen     # gerador com augmentation
val_test_datagen = pipeline.val_test_datagen  # gerador sem augmentation
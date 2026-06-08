# Diagnóstico de seleção de GPU no DirectML.
#
# Com tensorflow-directml-plugin a seleção de adaptador é feita pela variável
# de ambiente DML_VISIBLE_DEVICES, definida ANTES de importar o tensorflow.
# tf.config.set_visible_devices NÃO funciona com o dispositivo pluggable do
# DirectML (gera "AlreadyExistsError: device is being mapped to multiple
# devices" durante o fit).
#
# Rode este arquivo sozinho para confirmar qual adaptador o índice "1" seleciona
# antes de treinar. Ajuste o índice se a enumeração da sua máquina for diferente.
import os

os.environ.setdefault("DML_VISIBLE_DEVICES", "0")  # 0 = RTX 4050 (confirmado via gpu_probe.py)

import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("DML_VISIBLE_DEVICES:", os.environ.get("DML_VISIBLE_DEVICES"))

_gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs visíveis ao processo: {_gpus}")
if not _gpus:
    print("Nenhuma GPU visível — verifique o índice em DML_VISIBLE_DEVICES.")
else:
    # Sob DirectML, com a env var setada, só o adaptador escolhido aparece,
    # já reindexado como GPU:0. Um teste mínimo confirma execução na GPU.
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        print("Soma de teste na GPU:0 ->", (a + b).numpy())
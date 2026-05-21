import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = (224, 224)
NUM_CLASSES = 5

modelo_treinado = tf.keras.models.load_model("modelo_cobra_final.keras")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

modelo_site = tf.keras.Model(inputs, outputs)

# Copia os pesos das camadas treinadas, ignorando augmentation
pesos_treinados = modelo_treinado.get_weights()
modelo_site.set_weights(pesos_treinados)

modelo_site.save("modelo_site.h5")

print("Modelo para o site salvo como modelo_site.h5")
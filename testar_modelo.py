import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

MODELO = "modelo_cobra_final.keras"
IMAGEM_TESTE = "teste.jpeg"
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODELO)

class_names = [
    "cascavel",
    "coral",
    "jararaca",
    "nao_cobra",
    "outras_serpentes"
]

img = image.load_img(IMAGEM_TESTE, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)[0]

resultado = sorted(
    zip(class_names, pred),
    key=lambda x: x[1],
    reverse=True
)

print("\nResultado:")
for classe, prob in resultado:
    print(f"{classe}: {prob * 100:.2f}%")
import tensorflowjs as tfjs
import tensorflow as tf

modelo = tf.keras.models.load_model("modelo_cobra_final.keras")

tfjs.converters.save_keras_model(modelo, "modelo_tfjs")

print("Modelo convertido para TensorFlow.js com sucesso!")
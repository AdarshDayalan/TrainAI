import tensorflow as tf

model = tf.keras.models.load_model('NeuralNet/my_model')
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
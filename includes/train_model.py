import tensorflow as tf
import pandas as pd
import os 

from constants import label_names

dir_path = os.path.dirname(os.path.realpath("body_landmarks.csv"))
train_path = os.path.join(dir_path, "CSV", "body_landmarks.csv")

print(train_path)
# train_path = "C:/Users/adars/OneDrive/Documents/Adarsh Projects/final-project-yeswecan/CSV/body_landmarks.csv"
# train_path = "C:/Users/Basilio/Documents/yes we can project/final-project-yeswecan/CSV/body_landmarks.csv"
# test_path = 'CSV/test_data.csv'

train_data = pd.read_csv(train_path, delimiter=',', header=None)
train_data = train_data.sample(frac = 1)
 
train_x = train_data[range(1, 100)]
train_y = train_data[0]
 
train_input = tf.convert_to_tensor(train_x)
train_labels = tf.convert_to_tensor(train_y)
 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(99),
    tf.keras.layers.Dense(132),
    tf.keras.layers.Dense(33),
    tf.keras.layers.Dense(len(label_names)),
    tf.keras.layers.Dropout(rate=0.3)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_input, train_labels, epochs=25)
model.save('NeuralNet/my_model')
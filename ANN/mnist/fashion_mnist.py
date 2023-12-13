# 2023.12.12(M)  ____  12. December 2023 (Monday)

# Step 1: import
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Step 2: prepare dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

(image_train, label_train), (image_test, label_test) = fashion_mnist.load_data()
print("Train Image shape :", image_train.shape)
print("Train Label : ", label_train, "\n")
print(image_train[0])

# NUM = 20
# plt.figure(figsize=(15, 15))
# for idx in range(NUM):
#     sp = plt.subplot(5, 5, idx+1)
#     plt.imshow(image_train[idx])
#     plt.title(f'Label:{label_train[idx]}')
# plt.show()

# Step 3: build model
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(28, 28)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='tanh'))
model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Step 4: compile model
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 5: review model
model.summary()

# Step 6: train model
model.fit(image_train, label_train, epochs=16, batch_size=12)

# Step 7: evaluation model
model.evaluate(image_test, label_test)

# Step 8: save model
model.save('fashion_mnist.h5')

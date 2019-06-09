#%%
import tensorflow as tf
from tensorflow import keras

tf.__version__

#%%
keras.__version__

#%%
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

#%%
X_train_full.shape

#%%
X_train_full.dtype

#%%
X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

#%%
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
class_names[y_train[0]]

#%%
from keras.models import Sequential

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()

#%%
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

#%%
import pandas as pd

pd.DataFrame(history.history).plot(figsuze=8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

#%%
model.evaluate(X_test, y_test)

#%%
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

#%%
y_pred = model.predict_classes(X_new)
y_pred

#%%
import numpy as np

np.array(class_names)[y_pred]

#%%
y_new = y_test[:3]
y_new
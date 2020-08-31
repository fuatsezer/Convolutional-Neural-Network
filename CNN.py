import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
#%%
iris =load_iris()
X = iris.data[:,(2,3)]
y = (iris.target==0).astype(np.int)
per_clf = Perceptron()
per_clf.fit(X,y)
y_pred=per_clf.predict([[2,0.5]])
#%%
import tensorflow
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full,y_train_full),(X_test,y_test) = fashion_mnist.load_data()
#%%
print(X_train_full.shape)
print(y_train_full.shape)
#%%%
X_valid,X_train = X_train_full[:5000] / 255.0,X_train_full[5000:] / 255.0
y_valid,y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test/ 255.0
#%%
class_names = ["T-shirt/top","Trousers","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag",
               "Ankle boot"]
print(class_names[y_train[0]])
#%%
input_shape=(28,28,1)
#%%
X_train=X_train.reshape(55000,28,28,1)
X_valid = X_valid.reshape(5000,28,28,1)

#%% 
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64,(7,7),activation="relu",padding="same",input_shape=[28,28,1]))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(128,3,activation="relu",padding="same"))
model.add(keras.layers.Conv2D(128,3,activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(256,3,activation="relu",padding="same"))
model.add(keras.layers.Conv2D(256,3,activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64,activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10,activation="softmax"))
#%%
model.summary()
#%%
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
history = model.fit(X_train,y_train,epochs=30,validation_data=(X_valid,y_valid))

#%%
import pandas as pd 
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
#%%
X_test=X_test.reshape(10000,28,28,1)
model.evaluate(X_test,y_test)
#%% AlexNet
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(96,(11,11),activation="relu",padding="same",input_shape=[28,28,1]))
model.add(keras.layers.MaxPooling2D((3,3)))
model.add(keras.layers.Conv2D(256,3,activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D((3,3)))
model.add(keras.layers.Conv2D(384,3,activation="relu",padding="same"))
model.add(keras.layers.Conv2D(384,3,activation="relu",padding="same"))
model.add(keras.layers.Conv2D(256,3,activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D((3,3)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(4000,activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2000,activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10,activation="softmax"))
#%%
model.summary()
#%%
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
history = model.fit(X_train,y_train,epochs=30,validation_data=(X_valid,y_valid))
#%%
import pandas as pd 
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
#%%
X_test=X_test.reshape(10000,28,28,1)
model.evaluate(X_test,y_test)
#%% GoogleNet
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64,(7,7),activation="relu",padding="same",input_shape=[28,28,1]))
model.add(keras.layers.MaxPooling2D((3,3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64,1,activation="relu",padding="same"))
model.add(keras.layers.Conv2D(192,1,activation="relu",padding="same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((3,3)))
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10,activation="softmax"))
#%%
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
history = model.fit(X_train,y_train,epochs=30,validation_data=(X_valid,y_valid))
#%%
import pandas as pd 
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
#%%
X_test=X_test.reshape(10000,28,28,1)
model.evaluate(X_test,y_test)
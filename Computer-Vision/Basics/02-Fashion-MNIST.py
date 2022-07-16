# %%
import os
from datetime import datetime
import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# disable all debugging logs for Tensorflow
import logging
tf.get_logger().setLevel(logging.ERROR)
ADDRESS_MODELS = os.path.join("..", "..", "Models", "custom_models", "computer_vision")
TIME_NOW = datetime.now().strftime("%Y%m%d-%H%M%S")

# %% load data
label_classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = X_train.reshape(*X_train.shape, 1), X_test.reshape(*X_test.shape, 1)
X_train, X_test = X_train/255., X_test/255.
X_train, X_test = np.array(X_train, dtype=np.float32), np.array(X_test, dtype=np.float32)

# %%
N = 5
idx = [np.random.choice(range(len(X_train))) for _ in range(N)]
plt.figure(figsize=(15, 15))
for i, ix in enumerate(idx):
    plt.subplot(1, N, i+1)
    image = X_train[ix, :].reshape((28, 28))
    plt.imshow(image, cmap="gray")
    plt.title(label_classes[int(y_train[ix])])
    plt.axis('off')
plt.show()

# %% Create the Model
model = Sequential()
model.add(InputLayer(input_shape=X_train.shape[1:], name="Layer_input"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="SAME", activation="relu", name="Conv2D_1"))
model.add(BatchNormalization(name="BatchNorm_1"))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="SAME", activation="relu", name="Conv2D_2"))
model.add(BatchNormalization(name="BatchNorm_2"))
model.add(MaxPooling2D(pool_size=(8, 8), name="MaxPool_1"))
model.add(Dropout(rate=0.1, name="Dropout_1"))
model.add(Flatten(name="Flatten_1"))
model.add(Dense(units=16, activation="relu", name="Dense_1"))
model.add(Dense(units=len(label_classes), activation="softmax", name="Dense_output"))
model.summary()

# %%
optimizer = tf.optimizers.Adam(learning_rate=0.005)
metrics = [tf.metrics.SparseCategoricalAccuracy()]
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)

early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
learning_rate_reduction = ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.1)
checkpoint_best = ModelCheckpoint(
    filepath=os.path.join(ADDRESS_MODELS, "model_checkpoint_best", f"checkpoint_{TIME_NOW}"),
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    save_freq="epoch", verbose=1
)
tensorboard = TensorBoard(
    log_dir=os.path.join(ADDRESS_MODELS, "logs", TIME_NOW),
    write_graph=True,
    histogram_freq=1,
    write_images=True,
)
callbacks = [early_stopping, learning_rate_reduction, checkpoint_best, tensorboard]
# %%
model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    callbacks=callbacks,
)
# %%

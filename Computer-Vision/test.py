#%%
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.utils import to_categorical

# disable all debugging logs for Tensorflow
import logging
tf.get_logger().setLevel(logging.ERROR)

# %% Select the strategy
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    for i, gpu_device in enumerate(gpu_devices):
        device_details = tf.config.experimental.get_device_details(gpu_device)
        print(i+1, "- Device details:", device_details["device_name"], "- with Compute capability:", device_details["compute_capability"])
if tf.config.list_physical_devices("GPU"):
    strategy = tf.distribute.MirroredStrategy() # set to MirroredStrategy
    print("Strategy is set to MirroredStrategy")
else:
    strategy = tf.distribute.get_strategy() # set to the default strategy
    print("Strategy is set to DefaultDistributionStrategy")

#%% Load data
(X_train_, y_train_), (X_test_, y_test_) = mnist.load_data()
N_CATEGORIES = len(set(y_train_))
print(f"# of Categories: {N_CATEGORIES}")

# %% create the model
img = X_train_[0, :, :]


# %%

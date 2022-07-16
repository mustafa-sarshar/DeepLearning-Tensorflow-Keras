# %%
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv3D
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.utils import to_categorical
from skimage.data import coffee
import numpy as np
import matplotlib.pyplot as plt

# disable all debugging logs for Tensorflow
import logging
tf.get_logger().setLevel(logging.ERROR)

# %% create the model
img = coffee()
img = np.reshape(a=img, newshape=(1, *img.shape))

# %%
img = tf.convert_to_tensor(value=img, dtype=tf.float32)
img_conv_filt1 = Conv2D(filters=1, kernel_size=30, padding="same", input_shape=img.shape[1:])(img)
img_conv_filt3 = Conv2D(filters=3, kernel_size=30, padding="same", input_shape=img.shape[1:])(img)
img_conv_filt3_dil3 = Conv2D(filters=3, kernel_size=30, padding="same", dilation_rate=3, input_shape=img.shape[1:])(img)

# %%
fig, ax = plt.subplots(4)
ax[0].imshow(np.int16(img[0]))
ax[1].imshow(np.int16(img_conv_filt1[0]), cmap=plt.cm.gray)
ax[2].imshow(np.int16(img_conv_filt3[0]))
ax[3].imshow(np.int16(img_conv_filt3_dil3[0]))
plt.show();
# %%

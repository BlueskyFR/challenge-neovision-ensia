from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow.data import Dataset
from tensorflow._api.v2.data import Dataset
from tensorflow.keras import *
from tensorflow.keras.layers import *

print(f"Using Tensorflow {tf.__version__}")

import pathlib
import matplotlib.pyplot as plt
plt.style.use('dark_background')

# TensorFlow configuration
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Import TensorDash
# from tensordash.tensordash import Tensordash

# Load the TensorBoard notebook extension
log_dir = "logs/" + "bc=64"
tf.debugging.experimental.enable_dump_debug_info(log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
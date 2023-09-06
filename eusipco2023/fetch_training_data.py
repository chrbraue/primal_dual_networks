"""
fetch_training_data.py: Create TensorFlow Dataset from .wav files in IEEE corpus

Input: You can make modifications under (*). n and bitrate should be positive integers.
"""

# =============================================================================
# import python packages
# =============================================================================
import numpy as np
import os
import tensorflow as tf

# =============================================================================
# import own code
# =============================================================================
from auxiliary_functions import wav_folder_to_matrix, uniform_quantization, get_data

# =============================================================================
# authorship information
# =============================================================================
__author__ = 'Christoph Brauer'
__license__ = 'GPL'
__version__ = '1.0'
__maintainer__ = 'Christoph Brauer'
__email__ = 'christoph.brauer@dlr.de'
__status__ = 'Prototype'

# =============================================================================
# specify data split, frame length and bitrate (*)
# =============================================================================
n = 320
bitrate = 4

# =============================================================================
# load data from ../icassp2019/IEEE_corpus/ and create .tfrecord files
# =============================================================================

os.mkdir('IEEE_corpus')

# training data
path_source = f'../icassp2019/IEEE_corpus/train_data/'
Y = wav_folder_to_matrix(path_source, n)
X = uniform_quantization(Y, bitrate)
path_target = f'IEEE_corpus/ieee_speech_train_{bitrate}bit.tfrecord'
with tf.io.TFRecordWriter(path_target) as f:
    for i in range(Y.shape[1]):
        x_serialized = tf.io.serialize_tensor(X[:, i].astype(np.float32)).numpy()
        y_serialized = tf.io.serialize_tensor(Y[:, i].astype(np.float32)).numpy()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_serialized])),
                         'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_serialized]))
                         }
            )
        )
        f.write(example.SerializePartialToString())

# validation data
path_source = f'../icassp2019/IEEE_corpus/dev_data/'
Y = wav_folder_to_matrix(path_source, n)
X = uniform_quantization(Y, bitrate)
path_target = f'IEEE_corpus/ieee_speech_val_{bitrate}bit.tfrecord'
with tf.io.TFRecordWriter(path_target) as f:
    for i in range(Y.shape[1]):
        x_serialized = tf.io.serialize_tensor(X[:, i].astype(np.float32)).numpy()
        y_serialized = tf.io.serialize_tensor(Y[:, i].astype(np.float32)).numpy()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_serialized])),
                         'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_serialized]))
                         }
            )
        )
        f.write(example.SerializePartialToString())

# test data
path_source = f'../icassp2019/IEEE_corpus/dev_data/'
Y = wav_folder_to_matrix(path_source, n)
X = uniform_quantization(Y, bitrate)
path_target = f'IEEE_corpus/ieee_speech_test_{bitrate}bit.tfrecord'
with tf.io.TFRecordWriter(path_target) as f:
    for i in range(Y.shape[1]):
        x_serialized = tf.io.serialize_tensor(X[:, i].astype(np.float32)).numpy()
        y_serialized = tf.io.serialize_tensor(Y[:, i].astype(np.float32)).numpy()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_serialized])),
                         'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_serialized]))
                         }
            )
        )
        f.write(example.SerializePartialToString())

# now extract 1024 frames from each validation and test data that will be used subsequently
_, ds_val, ds_test, _ = get_data(bitrate=bitrate, batch_size=1024, repeat_data=1, shuffle_data=True)

[data] = [item for item in ds_val.take(1)]
np.savetxt('IEEE_corpus/inputs_val.txt', data[0])
np.savetxt('IEEE_corpus/outputs_val.txt', data[1])

[data] = [item for item in ds_test.take(1)]
np.savetxt('IEEE_corpus/inputs_test.txt', data[0])
np.savetxt('IEEE_corpus/outputs_test.txt', data[1])

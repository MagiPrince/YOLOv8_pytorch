import tensorflow as tf
import hls4ml
import os

os.environ['PATH'] += os.pathsep + '/tools/Xilinx/Vivado/2019.2/bin'

# Define a simple TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Generate the configuration dictionary from the TensorFlow model
cfg = hls4ml.utils.config.config_from_keras_model(model)

# Convert the TensorFlow model to an HLS implementation
hls_model = hls4ml.converters.keras_to_hls(cfg)

# Build the Vivado HLS project
hls_model.build(csim=False, synth=True, export=True)

import hls4ml
from hls4ml.converters import convert_from_keras_model
import os
import tensorflow as tf
import qkeras

os.environ['PATH'] += os.pathsep + '/tools/Xilinx/Vivado/2022.2/bin'

# Define the MLP architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=256, activation='relu', input_shape=(784,)),
    qkeras.layers.QDense(units=128, kernel_quantizer='quantized_bits(4,0,1)', bias_quantizer='quantized_bits(4,0,1)', activation='relu'),
    qkeras.layers.QDense(units=10, kernel_quantizer='quantized_bits(4,0,1)', bias_quantizer='quantized_bits(4,0,1)', activation='softmax')
])

# Compile the model with appropriate optimizer, loss function and metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the summary of the model
model.summary()

print("Here")

# Configure the conversion to Vitis HLS
config = hls4ml.utils.config_from_keras_model(model)
config['Backend'] = 'vivado'
config['BackendConfig'] = {'HLSConfig':{'Model':'vitis'}}

# Convert the model to Vitis HLS
hls_model = convert_from_keras_model(model, hls_config=config)

print("There")

# Save the generated code to a directory
output_dir = 'my_hls_project'
hls_model.compile()
hls_model.build(csim=False, synth=True, vsynth=True)
hls_model.write(output_dir)

# Define a simple TensorFlow model
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(28,28)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10)
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# # Generate the configuration dictionary from the TensorFlow model
# cfg = hls4ml.utils.config.config_from_keras_model(model)

# # Convert the TensorFlow model to an HLS implementation
# hls_model = hls4ml.converters.keras_to_hls(cfg)

# # Build the Vivado HLS project
# hls_model.build(csim=False, synth=True, export=True)

# Fetch a keras model from our example repository
# This will download our example model to your working directory and return an example configuration file
# config = hls4ml.utils.fetch_example_model('KERAS_3layer.json')

# You can print the configuration to see some default parameters
# print(config)

# Convert it to a hls project
# hls_model = hls4ml.converters.keras_to_hls(config)

# Print full list of example models if you want to explore more
# hls4ml.utils.fetch_example_list()

# Use Vivado HLS to synthesize the model
# This might take several minutes
# hls_model.build()

# Print out the report if you want
# hls4ml.report.read_vivado_report('my-hls-test')

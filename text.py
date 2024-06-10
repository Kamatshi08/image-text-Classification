import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image (replace 'path/to/your/image.jpg' with your actual image path)
image_path = "/mnt/c/Users/91892/Desktop/94603579ac87f7be7a37b4715a696ac3.png"
image = Image.open(image_path).convert("RGB")
input_shape = input_details[0]['shape']
image = image.resize((input_shape[1], input_shape[2]))  # Resize to expected shape
input_data = np.expand_dims(np.array(image), axis=0).astype(np.float32) / 255.0

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print the output (classification results)
print("Output:", output_data)

# To decode the output, you may need the labels file (e.g., labels_mobilenet_quant_v1_224.txt)
# You can download it from: https://storage.googleapis.com/download.tensorflow.org/models/tflite/labels_mobilenet_quant_v1_224.txt
# And then use the following code to decode the output:

labels_path = "path/to/labels_mobilenet_quant_v1_224.txt"
with open(labels_path, "r") as f:
    labels = f.readlines()

# Print the top-1 prediction
top_k = output_data[0].argsort()[-1:][::-1]
for i in top_k:
    print(f"{labels[i].strip()}: {output_data[0][i]}")

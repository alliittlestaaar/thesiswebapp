from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import cv2
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load YOLOv9 model once when the app starts
yolov9_session = ort.InferenceSession("varroayolov9.onnx")  # Ensure the path to your .onnx file is correct
input_name = yolov9_session.get_inputs()[0].name
output_name = yolov9_session.get_outputs()[0].name

# Load class names
with open("models/classes.txt", "r") as f:  # Update with the correct path to your classes file
    class_names = [line.strip() for line in f.readlines()]

def preprocess_image(image):
    # Target dimensions based on training setup
    target_height, target_width = 640, 640  # Ensure this matches your model's input size
    h, w, _ = image.shape

    # Calculate scale ratio to maintain aspect ratio
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image
    resized_img = cv2.resize(image, (new_w, new_h))

    # Create a blank canvas with the target dimensions (1120x640) and center the resized image
    padded_img = np.full((target_height, target_width, 3), 114, dtype=np.uint8)  # 114 is a neutral color
    padded_img[(target_height - new_h) // 2:(target_height - new_h) // 2 + new_h,
               (target_width - new_w) // 2:(target_width - new_w) // 2 + new_w] = resized_img

    # Normalize and convert to channels-first format
    padded_img = padded_img.astype(np.float32) / 255.0
    input_tensor = np.transpose(padded_img, (2, 0, 1))  # Change to channels-first
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension

    return input_tensor

# Postprocess function to handle model output (bounding boxes, confidence scores)
def postprocess(output, conf_threshold=0.25):
    predictions = output[0]  # Assuming output[0] contains the predictions
    results = []

    if len(predictions.shape) == 3:  # Check shape
        for prediction in predictions[0]:
            if len(prediction) >= 6:
                x, y, width, height, confidence, class_id = prediction[:6]

                if confidence >= conf_threshold:  # Apply confidence threshold
                    results.append({
                        "x": float(x),
                        "y": float(y),
                        "width": float(width),
                        "height": float(height),
                        "confidence": float(confidence),
                        "class_id": int(class_id),
                        "class_name": class_names[int(class_id)] if int(class_id) < len(class_names) else "Unknown"
                    })
    else:
        return {"error": "Unexpected output shape", "details": str(predictions)}

    return {"predictions": results}

# Route for YOLOv9 prediction
@app.route('/predict/yolov9', methods=['POST'])
def predict_yolov9():
    data = request.get_json()

    # Log the received data for debugging
    print("Received data:", data)

    # Check if the image was provided
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    # Decode the base64 image
    try:
        image_data = data['image']
        # Handle different image data formats
        if image_data.startswith('data:image/png;base64,'):
            image_data = image_data.replace('data:image/png;base64,', '')
        elif image_data.startswith('data:image/jpeg;base64,'):
            image_data = image_data.replace('data:image/jpeg;base64,', '')

        image_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Error decoding image:", e)
        return jsonify({"error": "Failed to decode image", "details": str(e)}), 400

    # Preprocess the image
    try:
        input_tensor = preprocess_image(img)
    except Exception as e:
        print("Error preprocessing image:", e)
        return jsonify({"error": "Failed to preprocess image", "details": str(e)}), 500

    # Run inference
    try:
        predictions = yolov9_session.run([output_name], {input_name: input_tensor})
        print("Raw predictions:", predictions)  # Debugging line
        results = postprocess(predictions)
    except Exception as e:
        print("Error during model inference:", e)
        return jsonify({"error": "Model inference failed", "details": str(e)}), 500

    # Log final results for debugging
    print("Processed results:", results)

    # Return the result
    return jsonify(results)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

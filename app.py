from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model("./mnist_digit_model.keras")


def preprocess_image(file_stream):
    """Preprocess image for MNIST digit prediction"""
    img = Image.open(file_stream).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img).astype("float32") / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input
    return img_array


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def hello():
        return {"message": "Hello from Flask!"}

    @app.route("/predict", methods=["POST"])
    def predict_digit():
        try:
            # Check if a file was uploaded
            if "image" not in request.files:
                return {"error": "No image part in the request"}, 400

            file = request.files["image"]
            if file.filename == "":
                return {"error": "No selected file"}, 400

            # Preprocess image
            img_array = preprocess_image(file)

            # Predict digit
            prediction = model.predict(img_array)
            predicted_digit = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][predicted_digit])

            return jsonify(
                {"predicted_digit": predicted_digit, "confidence": confidence}
            )

        except Exception as e:
            return {"error": str(e)}, 500

    return app


create_app()

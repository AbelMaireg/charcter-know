from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("./model/emnist_model.keras")

maps = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F",
    16: "G",
    17: "H",
    18: "I",
    19: "J",
    20: "K",
    21: "L",
    22: "M",
    23: "N",
    24: "O",
    25: "P",
    26: "Q",
    27: "R",
    28: "S",
    29: "T",
    30: "U",
    31: "V",
    32: "W",
    33: "X",
    34: "Y",
    35: "Z",
    36: "a",
    37: "b",
    38: "c",
    38: "d",
    39: "e",
    40: "f",
    41: "g",
    42: "h",
    43: "i",
    44: "j",
    45: "k",
    46: "l",
    47: "m",
    48: "n",
    49: "o",
    50: "p",
    51: "q",
    52: "r",
    53: "s",
    54: "t",
    55: "u",
    56: "v",
    57: "w",
    58: "x",
    59: "y",
    60: "z",
}


def preprocess_image(file_stream):
    """Preprocess image for MNIST digit prediction"""
    img = Image.open(file_stream).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img).astype("float32") / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input
    return img_array


def create_app():
    app = Flask(__name__)
    CORS(app)

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
            predicted_index = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][predicted_index])
            predicted_character = maps[predicted_index]

            print(f"Predicted index: {predicted_index}, Confidence: {confidence}")
            return jsonify(
                {"predicted_character": predicted_character, "confidence": confidence}
            )

        except Exception as e:
            return {"error": str(e)}, 500

    return app


create_app()

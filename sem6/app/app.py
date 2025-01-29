from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
import base64
from PIL import Image
import io

app = Flask(__name__)

# Load the CNN model
try:
    model = load_model('models/digit_recognition_cnn.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

def preprocess_image(image_data):
    # Remove the data URL prefix
    image_data = image_data.replace('data:image/png;base64,', '')

    # Decode base64 image
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # Preprocess the image
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to match model input size
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for model

    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        image_data = request.json['image']
        processed_image = preprocess_image(image_data)

        # Get prediction
        prediction = model.predict(processed_image)
        digit = np.argmax(prediction[0])
        confidence = float(prediction[0][digit])

        return jsonify({
            'digit': int(digit),
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

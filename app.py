from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = load_model('saved_models/model_cnn_v1.keras')

IMG_SIZE = 150

def prepare_image(img_file):
    # Open the image using PIL
    img = Image.open(img_file.stream).convert('RGB')  # Ensure image is in RGB format
    img = img.resize((IMG_SIZE, IMG_SIZE))  # Resize to (150, 150)
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Normalize the image
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['image']
        img = prepare_image(img_file)
        prediction = model.predict(img)[0][0]
        result = 'Pneumonia' if prediction > 0.5 else 'Normal'
        return render_template('results.html', prediction=result)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

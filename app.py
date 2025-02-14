from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as npp

app = Flask(__name__)

# Load the trained model
model = load_model('Glaucoma_detection_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_glaucoma(image_path):
    processed_img = preprocess_image(image_path)
    prediction = model.predict(processed_img)
    return prediction[0][0]

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/reviews')
def reviews():
    return render_template('reviews.html')

@app.route('/blogs')
def blogs():
    return render_template('blogs.html')

@app.route('/nav')
def nav():
    return render_template('nav.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('main.html', message='No file uploaded')
        
        file = request.files['file']
        
        # Check if the file is empty
        if file.filename == '':
            return render_template('main.html', message='No file selected')
        
        
        # Check if the file format is supported
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if file.filename.split('.')[-1].lower() not in allowed_extensions:
            return render_template('main.html', message='Unsupported file format')
        
        # Save the uploaded file
        file.save('static/uploads/uploaded_image.jpg')
        
        # Make predictions
        prediction = predict_glaucoma('static/uploads/uploaded_image.jpg')
        
        if prediction < 0.5:
            result = 'Glaucoma Not Detected'
        else:
            result = 'Glaucoma Detected'
        
        return render_template('main.html', message=result, prediction=prediction, image='static/uploads/uploaded_image.jpg')

if __name__ == '__main__':
    app.run(debug=True)




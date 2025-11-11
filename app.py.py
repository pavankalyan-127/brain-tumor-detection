# from flask import Flask, render_template, request, send_from_directory  (()) <main>  (())
# from tensorflow.keras.models import load_model
# from keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# import os

# # Initialize Flask app
# app = Flask(__name__)

# # Load the trained model
# model = load_model('model.keras')

# # Class labels
# class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# # Define the uploads folder
# UPLOAD_FOLDER = './uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Helper function to predict tumor type
# def predict_tumor(image_path):
#     IMAGE_SIZE = 128
#     img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#     img_array = img_to_array(img) / 255.0  # Normalize pixel values
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     predictions = model.predict(img_array)
#     predicted_class_index = np.argmax(predictions, axis=1)[0]
#     confidence_score = np.max(predictions, axis=1)[0]

#     if class_labels[predicted_class_index] == 'notumor':
#         return "No Tumor", confidence_score
#     else:
#         return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# # Route for the main page (index.html)
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Handle file upload
#         file = request.files['file']
#         if file:
#             # Save the file
#             file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(file_location)

#             # Predict the tumor
#             result, confidence = predict_tumor(file_location)

#             # Return result along with image path for display
#             return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f'/uploads/{file.filename}')

#     return render_template('index.html', result=None)

# # Route to serve uploaded files
# @app.route('/uploads/<filename>')
# def get_uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)


 ###(()) from chstgpt (()) 1
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
import os

# -----------------------------
# Initialize Flask app
# -----------------------------
app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------
# Load trained model
# -----------------------------
model = load_model('model.keras')
class_labels = ['pituitary', 'glioma', 'meningioma', 'notumor']

# -----------------------------
# Load CSV and compute one average cost
# -----------------------------
df = pd.read_csv("costcancer.csv")

# Clean column names
df.columns = [col.strip().replace(" ", "").replace("-", "").replace("(", "")
              .replace(")", "").replace("/", "").replace(".", "").replace("&", "and") 
              for col in df.columns]

# Rename columns
column_mapping = {
    'CostofCancerCarebyPhaseofCare': 'CancerSite',
    'Unnamed:1': 'Year',
    'Unnamed:2': 'Sex',
    'Unnamed:3': 'Age',
    'Unnamed:4': 'IncidenceandSurvivalAssumptions',
    'Unnamed:5': 'AnnualCostIncreaseAppliedtoinitialandlastphases',
    'Unnamed:6': 'TotalCosts',
    'Unnamed:7': 'InitialYearCost',
    'Unnamed:8': 'ContinuingPhaseCost',
    'Unnamed:9': 'LastYearCost'
}
df.rename(columns=column_mapping, inplace=True)

# Convert cost columns to numeric
for col in ['InitialYearCost', 'ContinuingPhaseCost', 'LastYearCost']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Compute ONE overall average
OVERALL_AVG_COST = int(df[['InitialYearCost', 'ContinuingPhaseCost', 'LastYearCost']].mean().mean())

# -----------------------------
# Predict tumor
# -----------------------------
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]
    tumor_type = class_labels[idx]

    if tumor_type == 'notumor':
        return "No Tumor", confidence, None
    else:
        treatment_cost = f"₹{OVERALL_AVG_COST:,}"
        return f"Tumor: {tumor_type}", confidence, treatment_cost

# -----------------------------
# Flask routes
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)
            result, confidence, cost = predict_tumor(path)
            return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%",
                                   treatment_cost=cost, file_path=f'/uploads/{file.filename}')
    return render_template('index.html', result=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

#  Brain Tumor Classification Using CNN (4 Classes)

This project focuses on classifying brain MRI images into **four types of tumor categories** using a Convolutional Neural Network (CNN). 
Brain tumor diagnosis through imaging is a crucial step for early detection and treatment planning.
The model is trained to accurately classify tumor types from MRI scans.

###  **Classes (4 Tumor Types)**

- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor (Healthy)**

##  Features

- ✔️ Custom CNN model trained on 4 tumor classes  
- ✔️ High validation accuracy  
- ✔️ Robust preprocessing for MRI scans  
- ✔️ Streamlit UI for image upload + live predictions  
- ✔️ Confidence score for each prediction  
- ✔️ Lightweight model, deployable on any machine  

##  Model Architecture (CNN)

The model consists of:

- Convolution layers (feature extraction)
- Max-pooling layers
- Batch normalization
- Dropout for regularization
- Fully connected dense layers
- Softmax classification head (4 classes)

Training includes augmentation:

- Rotation  
- Zoom  
- Horizontal/vertical shifts  


##  Dataset Summary

Dataset includes MRI scans categorized into:


All images were preprocessed to:

- Grayscale or RGB (depends on training)
- Normalized to 0–1
- Resized to (128×128 / 150×150)

##  Sample Output
 deployment link 
 https://braintumourdetection0.streamlit.app/

![image alt](https://github.com/pavankalyan-127/brain-tumor-detection/blob/main/brain_1.jpg?raw=true)
![image alt](https://github.com/pavankalyan-127/brain-tumor-detection/blob/main/brain_2.jpg?raw=true)
![image alt](https://github.com/pavankalyan-127/brain-tumor-detection/blob/main/brain_4.jpg?raw=true)
![image alt](https://github.com/pavankalyan-127/brain-tumor-detection/blob/main/brain_3.jpg?raw=true)

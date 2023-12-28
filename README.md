# Traffic Sign Classification

This project aims to develop a traffic sign detection model for classification purposes. The model can be used for real-time recognition of traffic signs, which can be particularly relevant for applications related to autonomous vehicles.

## Methodology

### 1. Dataset
The dataset used for training the model is sourced from the [German Traffic Sign Recognition Benchmark (GTSRB)]([https://benchmark.ini.rub.de/gtsrb_news.html](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)). It consists of images of traffic signs belonging to 43 different classes.

### 2. Model Architecture
The model is based on a Convolutional Neural Network (CNN) architecture. It includes convolutional layers, pooling layers, fully connected layers, and a softmax layer for classification.

### 3. Data Preprocessing
Images from the dataset are resized to 50x50 pixels to ensure consistent size. Additionally, pixel normalization is performed to scale values between 0 and 1.

### 4. Training
The model is trained on the dataset with cross-validation. The training process is monitored to evaluate accuracy and loss on both the training and validation sets.

### 5. Real-Time Detection
The application of the model for real-time traffic sign detection is demonstrated using the webcam. Results are displayed with predicted labels for each detected sign.

## Repository Contents

- **bicycles crossing.jpg**: Example of detection for the "bicycles crossing" class.
- **confusion_matrix.png**: Confusion matrix illustrating the model's performance.
- **loss_plot.png**: Graph showing the evolution of accuracy and loss during training.
- **speed limit 100.jpg**: Example of detection for the "Speed limit 100km/h" class.
- **Traffic_sign_classification_train.ipynb**: Code for training the model.
- **traffic_sign_model.h5**: Trained model saved in H5 format.
- **WebCam.ipynb**: Code for real-time detection using the webcam.
- **yield.jpg**: Example of detection for the "Yield" class.


Feel free to contribute and enhance this project!

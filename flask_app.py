import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model

# Load trained model
MODEL_PATH = 'traffic_sign_model.h5'
model = load_model(MODEL_PATH)

# Label mapping
LABELS = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
          'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
          'End of speed limit (80km/h)', 'Speed limit (100km/h)',
          'Speed limit (120km/h)', 'No passing',
          'No passing for vehicles over 3.5 metric tons',
          'Right-of-way at the next intersection', 'Priority road',
          'Yield', 'Stop', 'No vehicles',
          'Vehicles over 3.5 metric tons prohibited', 'No entry',
          'General caution', 'Dangerous curve to the left',
          'Dangerous curve to the right', 'Double curve', 'Bumpy road',
          'Slippery road', 'Road narrows on the right', 'Road work',
          'Traffic signals', 'Pedestrians', 'Children crossing',
          'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
          'End of all speed and passing limits', 'Turn right ahead',
          'Turn left ahead', 'Ahead only', 'Go straight or right',
          'Go straight or left', 'Keep right', 'Keep left',
          'Roundabout mandatory', 'End of no passing',
          'End of no passing by vehicles over 3.5 metric tons']

app = Flask(__name__)


# Initialize webcam
camera = cv2.VideoCapture(0)


def preprocess(frame):
    img = Image.fromarray(frame)
    img = img.resize((50, 50))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        img = preprocess(frame)
        preds = model.predict(img)
        label = LABELS[int(np.argmax(preds))]
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

import cv2
import numpy as np
from flask import Flask, Response, render_template, request, send_from_directory
from keras.models import load_model
import threading
import time
import os
from tensorflow.keras.utils import img_to_array
from mtcnn import MTCNN


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

app = Flask(__name__)

# Global variables for the video feed and threading control
video_thread_running = False
video_feed = None
video_thread = None

# Function to process the uploaded image
def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detector = MTCNN()  # Initialize MTCNN for face detection
    faces = detector.detect_faces(image)

    if faces:
        x, y, w, h = faces[0]['box']  # Get the bounding box for the first face
        face_crop = image[y:y + h, x:x + w]
        
        # Resize to match the input shape expected by the emotion model
        face_resized = cv2.resize(face_crop, (48, 48))
        
        # Convert to grayscale if needed
        if face_resized.shape[2] == 3:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)

        # Normalize and reshape for the model
        feature = img_to_array(face_resized).reshape((1, 48, 48, 1))
        feature = feature / 255.0
        return feature
    
    return None  # Return None if no face is detected

def generate_frames():
    global video_thread_running
    emotion_model = load_model('model.h5')
    video_feed = cv2.VideoCapture(0)  # Webcam index 0
    face_classifier = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    while video_thread_running:
        success, frame = video_feed.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5
        )

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray_resized = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray_resized.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_model.predict(roi)[0]
            emotion = emotion_labels[prediction.argmax()]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(
                frame,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    video_feed.release()
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed_route():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace;boundary=frame"
    )

@app.route("/start_video", methods=["POST"])
def start_video():
    global video_thread_running, video_thread

    # Only start if the thread isn't running
    if not video_thread_running:
        video_thread_running = True
        video_thread = threading.Thread(target=generate_frames)
        video_thread.start()
        print("Video thread started.")

    return "Video Started"

@app.route("/stop_video", methods=["POST"])
def stop_video():
    global video_thread_running

    if video_thread_running:
        video_thread_running = False
        time.sleep(1)  # Ensure the thread has time to stop
        print("Video thread stopped.")

    return "Video Stopped"

@app.route("/predict", methods=["POST"])
def predict():
    imagefile = request.files['imagefile']  
    upload_folder = 'static/uploads'
    os.makedirs(upload_folder, exist_ok=True)  
    image_filename = imagefile.filename
    image_path = os.path.join(upload_folder, image_filename)
    imagefile.save(image_path)  # Save the uploaded image

    # Process the image and get the features for prediction
    feature = process_image(image_path)

    if feature is None:
        # If no face is detected, inform the user
        classification = "No face detected in the image"
    else:
        emotion_model = load_model('model.h5')
        # Predict the emotion based on the features
        pred = emotion_model.predict(feature)
        pred_index = pred.argmax()  # Get the index of the maximum prediction
        
        pred_label = emotion_labels[pred_index]
        confidence = pred[0][pred_index]

        classification = f'{pred_label} ({confidence * 100:.2f}%)'  
    
    # Render the result on the webpage, displaying the prediction and the uploaded image
    return render_template("index.html", prediction=classification, image_path=image_filename)

@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    # Allow access to the uploaded files
    return send_from_directory("static/uploads", filename)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)

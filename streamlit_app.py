import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('model/best_model.keras')

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("😄 Face Emotion Detection App")

# Option selection
option = st.radio("Choose Input Type:", ["Upload Image", "Use Webcam"])

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# ================= IMAGE UPLOAD =================
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.reshape(face, (1, 48, 48, 1))

            prediction = model.predict(face, verbose=0)
            emotion = labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            text = f"{emotion} ({confidence:.2f})"

            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(image, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)

        st.image(image, channels="BGR")


# ================= WEBCAM =================
elif option == "Use Webcam":
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.reshape(face, (1, 48, 48, 1))

            prediction = model.predict(face, verbose=0)
            emotion = labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            text = f"{emotion} ({confidence:.2f})"

            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(image, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)

        st.image(image, channels="BGR")






        